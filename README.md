# User-Based Collaborative Filtering Recommender

This project implements a **user-based collaborative filtering system** using the [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/).
It predicts movie ratings for individual users and generates **group recommendations** based on multiple aggregation strategies.

---

## Features

- Automatically downloads and loads the MovieLens dataset
- Builds a user–item rating matrix
- Computes user–user similarity using:
  - **Pearson correlation**
  - **Spearman correlation with shrinkage adjustment**
- Predicts movie ratings using **k nearest neighbors**
- Supports **group recommendation methods**:
  - **Average** – mean of user predictions
  - **Least Misery** – prioritizes avoiding low ratings
  - **Consensus (mean − α·std)** – balances average preference and disagreement
  - **Least Disagreement** – ranks items with the smallest variance among group members
- Interactive **CLI interface** to explore predictions and group results

---

## Requirements

- Python 3.9+
- Install dependencies:
```
pip install pandas numpy requests
```

---

## Usage

Run the recommender directly:
```
python recommender.py
```

When launched, an interactive CLI opens:
```
=== Recommender CLI ===
p = toggle method (Pearson <-> Spearman+shrink)
1 = predict (userId + movieId) using SELECTED method
2 = group recommendations (Average) from seed user
3 = group recommendations (Least Misery) from seed user
q = quit
```

## How It Works

1. **Dataset Loading** – Ratings and movie metadata are loaded from MovieLens.
2. **Similarity Computation** – A pairwise correlation matrix between users is built.
3. **Prediction** – Unknown ratings are predicted using weighted averages of similar users.
4. **Group Aggregation** – Individual predictions are combined into group scores using the selected strategy.
5. **Consensus Method** – Balances group satisfaction and disagreement using `mean - α * std`.

The system stores similarity matrices for both Pearson and Spearman+shrink, allowing easy method switching.

---

## Adjustable Parameters
These can be adjusted in the .py-file as needed.
| Parameter    | Description                                         | Default |
|--------------|-----------------------------------------------------|---------|
| `GROUP_SIZE` | Number of users in a recommendation group           | `4`     |
| `MIN_COMMON` | Minimum number of co-rated movies for similarity    | `5`     |
| `ALPHA`      | Disagreement weighting in consensus method          | `0.6`   |
| `K`          | Number of nearest neighbors used in predictions     | `20`    |
| `TOP_N`      | Number of recommendations returned                  | `10`    |