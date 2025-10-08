# Complete-Python-3-Bootcamp
Course Files for Complete Python 3 Bootcamp Course on Udemy


Get it now for 95% off with the link:
https://www.udemy.com/complete-python-bootcamp/?couponCode=COMPLETE_GITHUB

Thanks!

## Loss Analysis Tool

The repository now includes `loss_analysis.py`, a utility script that helps
categorise autorater mistakes for web browsing agent models. Provide a CSV file
and a text file containing the loss categories, and the script will:

1. Filter the CSV to rows with a non-empty `issue_summary`.
2. Batch rows in groups of 100 and send them to an OpenAI chat model for
   categorisation.
3. Aggregate per-category counts for each model and update
   `loss_category_summary.csv`.
4. Generate a comparison chart saved to `loss_category_summary.png`.

Install the optional dependencies with `pip install openai pandas matplotlib`
before running the tool. Then execute `python loss_analysis.py --help` for the
full list of arguments. Use the `--dry-run` flag if you want to verify the
pipeline without calling the OpenAI API.
