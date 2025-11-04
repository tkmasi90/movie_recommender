import io
import os
import pandas as pd
import requests
from zipfile import ZipFile
import numpy as np

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "ml-latest-small")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")

GROUP_SIZE = 4

class DataLoader:
    def __init__(self):
        pass

    def read_ratings(self) -> pd.DataFrame:
        if not os.path.exists(RATINGS_PATH):
            self.fetch_data()
        return pd.read_csv(RATINGS_PATH)

    def read_movies(self) -> pd.DataFrame:
        if not os.path.exists(MOVIES_PATH):
            self.fetch_data()
        return pd.read_csv(MOVIES_PATH)

    def fetch_data(self):
        response = requests.get(URL)
        if response.status_code == 200:
            with ZipFile(io.BytesIO(response.content), 'r') as zip:
                zip.extractall(ROOT)
        else:
            raise Exception(f"Failed to download data, status code: {response.status_code}")

class RecommenderSystem:
    def __init__(self):
        self.loader = DataLoader()
        self.ratings_df = self.loader.read_ratings()
        self.movies_df = self.loader.read_movies()

        # rakennetaan perusmatriisi ja simit kerran
        self.ratings = self.build_user_item_matrix(self.ratings_df)
        self.min_common = 5
        self.sim_pearson, self.sim_spear_robust = self.user_user_similarity(self.ratings, self.min_common)

        # aktiivinen metodi (pearson | spearman)
        self.current_method = 'spearman'  # vaihda oletusta halutessasi

        self.current_group = pd.DataFrame()

    def build_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot ratings to user-item matrix with floats and NaNs for missing."""

        ratings = df.pivot(index="userId", columns="movieId", values="rating")
        return ratings.astype(float)

    def pairwise_common_counts(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """ Compute how many co-rated items each user pair has. """

        rated = ratings.notna().astype(np.int16)
        counts = rated @ rated.T
        counts.index = ratings.index
        counts.columns = ratings.index
        return counts

    def user_user_similarity(self, ratings: pd.DataFrame, min_common: int, lambda_: int = 30) -> tuple:
        """ Compute user-user correlations; also Spearman + shrinkage. """

        sim_pearson = ratings.T.corr(method="pearson",  min_periods=min_common)
        sim_spearman = ratings.T.corr(method="spearman", min_periods=min_common)

        n_uv = self.pairwise_common_counts(ratings).astype(float)
        n_uv = n_uv.reindex_like(sim_spearman)

        shrink = n_uv / (n_uv + float(lambda_))
        sim_spear_robust = sim_spearman * shrink
        return sim_pearson, sim_spear_robust
    
    def _active_sim(self) -> pd.DataFrame:
        """ Return the currently selected similarity matrix. """
        return self.sim_pearson if self.current_method == 'pearson' else self.sim_spear_robust

    def predict_scores(self, user: int, movie: int, sim: pd.DataFrame, ratings: pd.DataFrame, k: int) -> float:
        """ Predict score for user and movie based on k most similar users. """
        if movie in ratings.columns and not np.isnan(ratings.at[user, movie]):
            return ratings.at[user, movie]

        user_mean = ratings.loc[user].mean(skipna=True)
        if np.isnan(user_mean):
            user_mean = ratings.stack().mean()

        if movie not in ratings.columns:
            return user_mean

        users_rated = ratings[movie]
        neighbors = users_rated[users_rated.notna()].index.drop(user, errors='ignore')
        if len(neighbors) == 0:
            return user_mean

        sims = sim.loc[user, neighbors]
        top = sims.abs().sort_values(ascending=False).head(k).index
        sims_top = sims.loc[top]
        r_b_p = ratings.loc[top, movie].astype(float)
        r_b_hat = ratings.loc[top].mean(axis=1, skipna=True).astype(float)

        num = (sims_top * (r_b_p - r_b_hat)).sum()
        den = sims_top.abs().sum()
        if den == 0 or pd.isna(num):
            return user_mean
        return float(user_mean + num / den)

    def topk_neighbors_group(self, seed_user: int, sim: pd.DataFrame, k: int = GROUP_SIZE-1) -> list:
        """ Get top-k similar users to seed_user. """

        s = sim.loc[seed_user].drop(seed_user, errors='ignore').dropna()
        top = s.sort_values(ascending=False).head(k).index.tolist()
        return [seed_user] + top
    
    def _initialize_group(self, group_users, ratings, sim, k=20) -> pd.DataFrame:
        """  """
        seen_by_group = set()
        for u in group_users:
            seen_by_group |= set(ratings.columns[ratings.loc[u].notna()])

        candidates = [i for i in ratings.columns if i not in seen_by_group]
        if not candidates:
            return pd.DataFrame()

        per_user_scores = {}
        for u in group_users:
            preds = {}
            for i in candidates:
                preds[i] = self.predict_scores(u, i, sim, ratings, k)
            per_user_scores[u] = pd.Series(preds)

        return pd.DataFrame(per_user_scores)

    def group_recs_average(self, group_users, ratings, sim, topn: int = 10) -> pd.Series:
        """ Generate group recommendations using average strategy. """

        M = self._initialize_group(group_users, ratings, sim, k=20)
        self.current_group = M
        g_scores = M.mean(axis=1).sort_values(ascending=False)
        return g_scores.head(topn)

    def group_recs_least_misery(self, group_users, ratings, sim, topn: int = 10) -> pd.Series:
        """ Generate group recommendations using least misery strategy. """

        M = self._initialize_group(group_users, ratings, sim, k=20)
        self.current_group = M
        g_scores = M.min(axis=1).sort_values(ascending=False)
        return g_scores.head(topn)

    def group_recs_consensus(self, topn: int = 10, alpha: float = 0.6) -> pd.Series:
        """ Generate group recommendations based on consensus (mean - alpha * std). """

        M = self.current_group
        means = M.mean(axis=1)
        std = M.std(axis=1, ddof=0)
        g_scores = (means - alpha * std).sort_values(ascending=False)
        return g_scores.head(topn)
    
    def rank_by_lowest_disagreement(self, movie_ids: list, topn: int = 10) -> pd.Series:
        """ Rank given movie IDs by lowest disagreement score. """

        M = self.current_group.loc[movie_ids]
        std = M.std(axis=1, ddof=0)
        g_scores = std.sort_values(ascending=True)
        return g_scores.head(topn)

    def get_movie_name(self, movieId: int) -> str:
        """ Get movie title by movieId. """

        row = self.movies_df[self.movies_df['movieId'] == movieId]
        return row.iloc[0]['title'] if not row.empty else "Unknown Movie"

    # --- INTERACTIVE CLI ---

    def cli(self):
        print("\n=== Recommender CLI ===")
        print("p = toggle method (Pearson <-> Spearman+shrink)")
        print("1 = predict (userId + movieId) using SELECTED method")
        print("2 = group recommendations (Average) from seed user using SELECTED method")
        print("3 = group recommendations (Least Misery) from seed user using SELECTED method")
        print("q = quit\n")

        def method_label():
            return "PEARSON" if self.current_method == 'pearson' else "SPEARMAN+shrink"

        while True:
            cmd = input(f"[method={method_label()}] Select [p/1/2/q]: ").strip().lower()
            if cmd == 'q':
                print("Bye.")
                break

            elif cmd == 'p':
                # toggle
                self.current_method = 'pearson' if self.current_method != 'pearson' else 'spearman'
                print(f"-> Method switched to {method_label()}\n")

            elif cmd == '1':
                try:
                    u = int(input(" userId: ").strip())
                    m = int(input(" movieId: ").strip())
                    sim = self._active_sim()
                    score = self.predict_scores(u, m, sim, self.ratings, k=20)
                    print(f"Predicted ({method_label()}) u={u}, m={m} ({self.get_movie_name(m)}): {score:.2f}\n")
                except Exception as e:
                    print(f"Error: {e}\n")

            elif cmd == '2':
                try:
                    seed = int(input(" seed userId: (0 for random group): ").strip())
                    sim = self._active_sim()
                    if seed == 0:
                        group = np.random.choice(a=self.ratings.index.tolist(), size=GROUP_SIZE, replace=False).tolist()
                    else:
                        group = self.topk_neighbors_group(seed_user=seed, sim=sim, k=3)
                    print(f" Group users (seed={seed}, method={method_label()}): {group}")
                    print()
                    recs = self.group_recs_average(group, self.ratings, sim, topn=10)
                    if recs.empty:
                        print(" No candidates found.\n")
                        continue
                    print(" Top-10 (AVERAGE):")
                    for mid, sc in recs.items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    print(" LEAST DISAGREEMENT (TOP-10):")
                    for mid, sc in self.rank_by_lowest_disagreement(recs.index.tolist()).items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    consensus_recs = self.group_recs_consensus(topn=10)
                    if consensus_recs.empty:
                        print(" No candidates found for disagreement.\n")
                        continue
                    print(" CONSENSUS (ALL MOVIES):")
                    for mid, sc in consensus_recs.items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    
                except Exception as e:
                    print(f"Error: {e}\n")

            elif cmd == '3':
                try:
                    seed = int(input(" seed userId: (0 for random group): ").strip())
                    sim = self._active_sim()
                    if seed == 0:
                        group = np.random.choice(a=self.ratings.index.tolist(), size=GROUP_SIZE, replace=False).tolist()
                    else:
                        group = self.topk_neighbors_group(seed_user=seed, sim=sim, k=3)
                    print(f" Group users (seed={seed}, method={method_label()}): {group}")
                    print()
                    recs = self.group_recs_least_misery(group, self.ratings, sim, topn=10)
                    if recs.empty:
                        print(" No candidates found.\n")
                        continue
                    print(" Top-10 (LEAST MISERY):")
                    for mid, sc in recs.items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    print(" LEAST DISAGREEMENT (TOP-10):")
                    for mid, sc in self.rank_by_lowest_disagreement(recs.index.tolist()).items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    consensus_recs = self.group_recs_consensus(topn=10)
                    if consensus_recs.empty:
                        print(" No candidates found for disagreement.\n")
                        continue
                    print(" CONSENSUS (ALL MOVIES):")
                    for mid, sc in consensus_recs.items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()

                except Exception as e:
                    print(f"Error: {e}\n")

            else:
                print("Unknown command.\n")

if __name__ == "__main__":
    recommender = RecommenderSystem()
    print(f"Dataset rows: {recommender.ratings_df.shape[0]}, columns: {recommender.ratings_df.shape[1]}")
    print(f"Users: {len(recommender.ratings.index)}, Movies: {len(recommender.ratings.columns)}\n")
    recommender.cli()