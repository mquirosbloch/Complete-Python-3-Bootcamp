"""Loss analysis categorisation pipeline.

This script processes CSV exports from the Web Browsing agent autorater and
classifies every row with an ``issue_summary`` using an OpenAI chat model. The
results are aggregated per-model and visualised to support loss analysis.

Usage example::

    python loss_analysis.py \
        --csv ./data/model_a.csv \
        --model-name model-a \
        --categories categories.txt \
        --summary-table loss_category_summary.csv \
        --chart loss_category_summary.png

The ``categories`` file should contain one category per line. Optionally a
description can follow the category name, separated by a ``|`` character.

The script supports a ``--dry-run`` mode for testing without performing OpenAI
API calls. In this mode every row is labelled with ``UNCLASSIFIED``.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

try:  # pragma: no cover - optional dependency when running with --dry-run.
    import openai
except ImportError:  # pragma: no cover - importing lazily prevents hard failure.
    openai = None  # type: ignore


BATCH_SIZE = 100


@dataclass(frozen=True)
class Category:
    """Represents a loss category with an optional description."""

    name: str
    description: str | None = None

    @classmethod
    def from_line(cls, line: str) -> "Category":
        name, *rest = line.split("|", maxsplit=1)
        name = name.strip()
        if not name:
            raise ValueError("Category names cannot be empty.")
        description = rest[0].strip() if rest else None
        return cls(name=name, description=description or None)


def load_categories(path: Path) -> List[Category]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError(f"No categories found in {path}.")
    return [Category.from_line(line) for line in lines]


def read_loss_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_columns = {
        "task_id",
        "step_index",
        "website_issue",
        "screenshot_description_correct",
        "thought_reasonable",
        "action_matches_thought",
        "incorrect_coordinates",
        "issue_summary",
        "autorater_failure",
        "g_lab_url",
    }
    missing = expected_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {sorted(missing)} in {path}.")
    df = df.copy()
    df["issue_summary"] = df["issue_summary"].fillna("").astype(str)
    filtered = df[df["issue_summary"].str.strip() != ""].reset_index(drop=True)
    return filtered


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def build_prompt(categories: Sequence[Category], rows: pd.DataFrame) -> str:
    category_lines = [
        f"- {cat.name}: {cat.description}" if cat.description else f"- {cat.name}"
        for cat in categories
    ]
    rows_payload = rows[
        [
            "task_id",
            "step_index",
            "website_issue",
            "screenshot_description_correct",
            "thought_reasonable",
            "action_matches_thought",
            "incorrect_coordinates",
            "issue_summary",
            "autorater_failure",
            "g_lab_url",
        ]
    ]
    serialised_rows = json.dumps(rows_payload.to_dict(orient="records"), ensure_ascii=False)
    prompt = textwrap.dedent(
        f"""
        You are an expert analyst labelling loss types for a web browsing agent.
        You will receive {len(rows)} log entries, each corresponding to an agent
        step that contained an issue. For each entry choose exactly one category
        from the list provided below and return a JSON array with {len(rows)}
        objects in the same order. Each object must contain the keys
        "task_id", "step_index", "category", and "explanation". The
        explanation should be a short (<=20 word) reason for your choice.

        Categories:
        {os.linesep.join(category_lines)}

        Input rows (JSON array):
        {serialised_rows}
        """
    ).strip()
    return prompt


def call_openai(model: str, prompt: str) -> str:
    if openai is None:
        raise RuntimeError(
            "openai package is not installed. Install openai>=0.27.0 or run with --dry-run."
        )
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You classify agent errors into predefined categories.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"]


def parse_model_output(text: str, expected_len: int) -> List[Dict[str, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(data, list) or len(data) != expected_len:
        raise ValueError(
            f"Expected JSON array of length {expected_len}, received {type(data)} with length {len(data) if isinstance(data, list) else 'n/a'}."
        )
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item in the response must be an object.")
        for key in ("task_id", "step_index", "category", "explanation"):
            if key not in item:
                raise ValueError(f"Missing key '{key}' in response item: {item}")
    return data


def label_rows(
    df: pd.DataFrame,
    categories: Sequence[Category],
    model: str,
    dry_run: bool = False,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for chunk in chunk_dataframe(df, BATCH_SIZE):
        if dry_run:
            for _, row in chunk.iterrows():
                results.append(
                    {
                        "task_id": str(row["task_id"]),
                        "step_index": int(row["step_index"]),
                        "category": "UNCLASSIFIED",
                        "explanation": "Dry run placeholder",
                    }
                )
            continue

        prompt = build_prompt(categories, chunk)
        raw_output = call_openai(model=model, prompt=prompt)
        parsed = parse_model_output(raw_output, len(chunk))
        results.extend(parsed)
    return results


def update_summary_table(
    summary_path: Path,
    model_name: str,
    categories: Sequence[Category],
    classifications: Sequence[Dict[str, str]],
) -> pd.DataFrame:
    category_names = [cat.name for cat in categories]
    counts = {name: 0 for name in category_names}

    for entry in classifications:
        category = entry["category"]
        if category not in counts:
            counts.setdefault(category, 0)
        counts[category] += 1

    summary_df = pd.DataFrame([counts], index=[model_name])

    if summary_path.exists():
        existing = pd.read_csv(summary_path, index_col=0)
        combined = existing.combine_first(summary_df)
        combined.loc[model_name] = summary_df.iloc[0]
        combined = combined.fillna(0).astype(int).sort_index()
    else:
        combined = summary_df.fillna(0).astype(int)

    combined.to_csv(summary_path)
    return combined


def plot_summary_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    ax = summary_df.sort_index().plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Count of mistakes")
    ax.set_xlabel("Model")
    ax.set_title("Loss category counts per model")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(ax.figure)


def save_classifications(path: Path, classifications: Sequence[Dict[str, str]]) -> None:
    df = pd.DataFrame(classifications)
    df.to_csv(path, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorise web browsing agent losses.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the loss CSV file.")
    parser.add_argument("--model-name", required=True, help="Identifier for the model being analysed.")
    parser.add_argument(
        "--categories",
        type=Path,
        required=True,
        help="Path to a text file containing loss categories.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model name to use for classification.",
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=Path("loss_category_summary.csv"),
        help="Path where the aggregated summary table will be stored.",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("loss_category_summary.png"),
        help="Path to save the comparison chart.",
    )
    parser.add_argument(
        "--classified-output",
        type=Path,
        help="Optional path to save the row-level classification results as CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI calls and emit placeholder classifications.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    categories = load_categories(args.categories)
    df = read_loss_csv(args.csv)
    if df.empty:
        raise SystemExit("No issue_summary rows found in the provided CSV.")

    classifications = label_rows(
        df=df,
        categories=categories,
        model=args.openai_model,
        dry_run=args.dry_run,
    )

    if args.classified_output:
        save_classifications(args.classified_output, classifications)

    summary_df = update_summary_table(
        summary_path=args.summary_table,
        model_name=args.model_name,
        categories=categories,
        classifications=classifications,
    )

    plot_summary_table(summary_df, args.chart)


if __name__ == "__main__":
    main()

