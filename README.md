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
  - **SDAA (Sequential Dynamic Aggregation Algorithm)** – multi-round group recommendation process with dynamic α updates based on user satisfaction
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
4 = run SDAA session from seed user using SELECTED method
q = quit
```

## How It Works

1. **Dataset Loading** – Ratings and movie metadata are loaded from MovieLens.
2. **Similarity Computation** – A pairwise correlation matrix between users is built.
3. **Prediction** – Unknown ratings are predicted using weighted averages of similar users.
4. **Group Aggregation** – Individual predictions are combined into group scores using the selected strategy.
5. **Consensus Method** – Balances group satisfaction and disagreement using `mean - α * std`.
6. **SDAA**: An iterative group recommendation method that updates disagreement (α) each round based on user satisfaction to gradually refine a shared set of top items.

The system stores similarity matrices for both Pearson and Spearman+shrink, allowing easy method switching.

---

## Adjustable Parameters
These can be adjusted in the `.py` file as needed.

| Parameter        | Description                                                             | Default |
|------------------|-------------------------------------------------------------------------|---------|
| `GROUP_SIZE`     | Number of users in a recommendation group                               | `4`     |
| `MIN_COMMON`     | Minimum number of co-rated movies for similarity                        | `15`    |
| `ALPHA`          | Disagreement weighting in consensus method                              | `0.6`   |
| `K`              | Number of nearest neighbors used in predictions                         | `20`    |
| `TOP_N`          | Number of recommendations returned                                      | `10`    |
| `TAU`            | Disagreement threshold used inside SDAA                                 | `0.3`   |
| `USE_EXCLUSION`  | Whether SDAA removes previously selected items from later iterations    | `True`  |


## Notes

### Group Selection
- Providing a **valid user ID** in group recommendation mode makes the system
  simulate a *friend group* by selecting that user’s **k-nearest neighbors**.
- To form a **completely random group**, use user ID `0`.  
  In this case, similarity is ignored and users are sampled uniformly at random.

### Exclusion Behavior in Sequential Recommendations
The parameter `USE_EXCLUSION` determines whether items chosen in earlier SDAA
iterations are removed from future rounds.

- **`USE_EXCLUSION = True`**  
  The algorithm constructs a *multi-step recommendation set*.  
  Once an item is selected, it is not considered again, encouraging **diversity**
  across iterations.

- **`USE_EXCLUSION = False`**  
  All items remain available in every round.  
  Strong recommendations can appear repeatedly, useful when **stability** and
  consistent top choices are preferred over diversity.

---

*Note: Portions of this README were generated with the help of GenAI.*
