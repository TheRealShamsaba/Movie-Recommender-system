# Movie Recommender System

An end‑to‑end recommender built on the MovieLens 100k dataset. It explores descriptive statistics, content filtering, collaborative filtering, and a KNN accuracy workflow that tunes hyper‑parameters with cross‑validation and reports RMSE/MAE along with visual diagnostics.

## Project Structure

- `recommend.py` – main analysis script/notebook export covering EDA, similarity search, KNN evaluation, and visualization.
- `app.py` + `templates/` – starter Flask interface (optional) for serving recommendations.
- `movies.csv`, `ratings.csv`, `ml-100k/` – MovieLens data files (unzipped locally).
- `Untitled-1.ipynb` – scratch notebook (not required to run).

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn fuzzywuzzy python-levenshtein
```

## Running the Analysis

```bash
python3 recommend.py
```

The script will:

1. Split MovieLens ratings into train and test sets.
2. Build sparse matrices/mappings.
3. Compare a user-mean baseline against item-based KNN predictions.
4. Perform 5-fold cross-validation over multiple `k` values and distance metrics.
5. Save CV results to `knn_cv_results.csv`, print the top configurations, and report final RMSE/MAE + improvements over the baseline.
6. Plot predicted-vs-actual ratings and error distributions for inclusion in reports.
7. Print tuned recommendations for a sample movie (ID 1 by default).

You can import `run_accuracy_pipeline()` from `recommend.py` to reuse the evaluation flow in notebooks without re-running earlier exploratory cells.

## Using the (Optional) Flask App

1. Ensure the `recommend.py` script has been executed at least once so matrices/mappers exist or adapt the app to load them.
2. Run `python3 app.py`.
3. Visit `http://localhost:5000` to search for movies and view recommendations.

## Notes for Reports

- Reference `knn_cv_results.csv` for detailed RMSE/MAE tables.
- Include the generated plots to illustrate prediction quality.
- Mention the baseline vs tuned metrics to highlight the accuracy gains your professor requested.

## License

MovieLens data is provided by GroupLens (see `ml-100k/u.info`). All project code is available for educational use.***
