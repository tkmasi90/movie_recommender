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

GROUP_SIZE = 4 # default group size for recommendations
MIN_COMMON = 15  # minimum common ratings for similarity
ALPHA = 0.6  # consensus parameter
K = 20  # number of neighbors for predictions
TOP_N = 10  # number of recommendations to return

TAU = 0.3  # disagreement threshold for SDAA
USE_EXCLUSION = True

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

        # build user-item matrix and similarities
        self.ratings = self.build_user_item_matrix(self.ratings_df)
        self.min_common = MIN_COMMON
        self.sim_pearson, self.sim_spear_robust = self.user_user_similarity(self.ratings, self.min_common)

        # active method (pearson | spearman)
        self.current_method = 'spearman' # default

        # current group scores
        self.current_group_preds = pd.DataFrame()
        self.previous_group_preds = pd.DataFrame()
        self._pred_cache = {}

        self.topn = TOP_N

    def build_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item matrix: rows = users, columns = movies, values = ratings."""
        ratings = df.pivot(index="userId", columns="movieId", values="rating")
        return ratings.astype(float)

    def pairwise_common_counts(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Count how many movies each user pair has rated in common."""
        rated = ratings.notna().astype(np.int16)
        counts = rated @ rated.T
        counts.index = ratings.index
        counts.columns = ratings.index
        return counts

    def user_user_similarity(self, ratings: pd.DataFrame, min_common: int, lambda_: int = 30) -> tuple:
        """Compute user-user similarity (Pearson and Spearman with shrinkage)."""
        sim_pearson = ratings.T.corr(method="pearson",  min_periods=min_common)
        sim_spearman = ratings.T.corr(method="spearman", min_periods=min_common)

        u_u_matrix = self.pairwise_common_counts(ratings).astype(float)

        # Apply shrinkage to Spearman similarity
        shrink = u_u_matrix / (u_u_matrix + float(lambda_))
        sim_spear_robust = sim_spearman * shrink
        return sim_pearson, sim_spear_robust
    
    def _active_sim(self) -> pd.DataFrame:
        """ Return the currently selected similarity matrix. """
        return self.sim_pearson if self.current_method == 'pearson' else self.sim_spear_robust

    def predict_score(self, user: int, movie: int, sim: pd.DataFrame, ratings: pd.DataFrame, k: int) -> float:
        """Predict a user's rating for a movie using k most similar users."""
        # If user has already rated the movie, return that rating
        if movie in ratings.columns and not np.isnan(ratings.at[user, movie]):
            return ratings.at[user, movie]

        # Calculate user mean
        user_mean = ratings.loc[user].mean(skipna=True)
        if np.isnan(user_mean):
            user_mean = ratings.stack().mean()

        # If movie not rated by anyone, return user mean
        if movie not in ratings.columns:
            return user_mean

        # Find users who rated the movie
        users_rated = ratings[movie]
        neighbors = users_rated[users_rated.notna()].index.drop(user, errors='ignore')
        if len(neighbors) == 0:
            return user_mean

        # Get similarities and select top-k
        sims = sim.loc[user, neighbors]
        top = sims.abs().sort_values(ascending=False).head(k).index
        sims_top = sims.loc[top]

        # Compute prediction
        r_b_p = ratings.loc[top, movie].astype(float) # ratings by top neighbors
        r_b_hat = ratings.loc[top].mean(axis=1, skipna=True).astype(float) # their means

        # Calculate weighted sum
        num = (sims_top * (r_b_p - r_b_hat)).sum() # numerator
        den = sims_top.abs().sum() # denominator
        if den == 0 or pd.isna(num):
            return user_mean
        return float(user_mean + num / den)
    
    def predict_score_cached(self, user: int, movie: int, sim: pd.DataFrame, ratings: pd.DataFrame, k: int) -> float:
        """
        Return a predicted rating p_j(u, i) for a given user-movie-method.
        Uses a cache to avoid recomputing predictions that have already been calculated earlier during the same session.
        """
        key = (user, movie, self.current_method)
        v = self._pred_cache.get(key)
        if v is None:
            v = self.predict_score(user, movie, sim, ratings, k)
            self._pred_cache[key] = v
        return v
    
    def user_recommendations(self, user: int, ratings: pd.DataFrame, sim: pd.DataFrame,
                         top_k: int = TOP_N, exclude: set | None = None) -> pd.Series:
        """
        Generate the individual top-k recommendation list A(u, j) for user u.
        """
        exclude = exclude or set()
        seen_by_user = set(ratings.columns[ratings.loc[user].notna()])
        candidates = [i for i in ratings.columns if i not in seen_by_user and i not in exclude]
        if not candidates:
            return pd.Series(dtype=float)
        preds = {i: self.predict_score_cached(user, i, sim, ratings, K) for i in candidates}  # <- K
        return pd.Series(preds, dtype=float).sort_values(ascending=False).head(top_k)


    def topk_neighbors_group(self, seed_user: int, sim: pd.DataFrame, k: int = GROUP_SIZE-1) -> list:
        """Return a group consisting of the seed user and their k most similar users."""

        s = sim.loc[seed_user].drop(seed_user, errors='ignore').dropna()
        top = s.sort_values(ascending=False).head(k).index.tolist()
        return [seed_user] + top
    
    def _initialize_group(self, group_users, ratings, sim, k=K, movies=None) -> pd.DataFrame:
        """Compute predicted scores for all candidate movies for each user in the group."""

        # Identify candidate movies (not yet rated by any group member)
        seen_by_group = set()
        for u in group_users:
            seen_by_group |= set(ratings.columns[ratings.loc[u].notna()])

        candidates = set()
        if movies is not None:
            candidates = [i for i in movies if i not in seen_by_group]
        else:
            candidates = [i for i in ratings.columns if i not in seen_by_group]
        if not candidates:
            return pd.DataFrame()

        # Predict scores for each user in the group for candidate movies
        per_user_scores = {}
        for u in group_users:
            preds = {}
            for i in candidates:
                preds[i] = self.predict_score_cached(u, i, sim, ratings, k)
            per_user_scores[u] = pd.Series(preds)

        return pd.DataFrame(per_user_scores)

    def _average(self, M) -> pd.Series:
        """Generate group recommendations using the Average strategy."""

        g_scores = M.mean(axis=1).sort_values(ascending=False)
        return g_scores.head(self.topn)

    def _least_misery(self, M) -> pd.Series:
        """Generate group recommendations using the Least Misery strategy."""

        g_scores = M.min(axis=1).sort_values(ascending=False)
        return g_scores.head(self.topn)
    
    def group_recs(self, ratings, sim, group_users, method, k=K) -> pd.Series:
        M = self._initialize_group(group_users, ratings, sim, k=k)
        self.current_group_preds = M
        if method == "average":
            return self._average(M)
        elif method == "least_misery":
            return self._least_misery(M)

    def rank_by_lowest_disagreement(self, movie_ids: list) -> pd.Series:
        """Rank movies by lowest standard deviation of group ratings (least disagreement)."""

        M = self.current_group_preds.loc[movie_ids]
        std = M.std(axis=1, ddof=0)
        # Rank by lowest disagreement
        g_scores = std.sort_values(ascending=True)
        return g_scores.head(self.topn)

    def group_recs_consensus(self, alpha: float = ALPHA) -> pd.Series:
        """Generate group recommendations using Consensus method (mean - alpha * std)."""

        M = self.current_group_preds
        means = M.mean(axis=1)
        # Calculate standard deviation for each movie. Measures how much group members disagree.
        std = M.std(axis=1, ddof=0)
        # Compute a consensus score per movie
        g_scores = (means - alpha * std).sort_values(ascending=False)
        return g_scores.head(self.topn)

    def get_movie_name(self, movieId: int) -> str:
        """ Get movie title by movieId. """

        row = self.movies_df[self.movies_df['movieId'] == movieId]
        return row.iloc[0]['title'] if not row.empty else "Unknown Movie"
    
    # ---- Sequential Group Recommendations ----

    def _sdaa_scores(self, M: pd.DataFrame, alpha: float, user_weights: dict | None = None) -> pd.Series:
        """ Compute the SW-SDAA aggregation score for all candidate movies in Gl.

        If user_weights is provided, compute a weighted average where users with
        lower past satisfaction get higher weight.
        """
        if user_weights is None:
            avg = M.mean(axis=1)
        else:
            w = pd.Series(user_weights, dtype=float)
            w = w / w.sum()  # normalize to 1
            avg = (M * w).sum(axis=1)

        least = M.min(axis=1)
        return (1.0 - alpha) * avg + alpha * least


    def _user_satisfaction(self, M, group, gr_j, A):
        """ Compute individual user satisfaction sat(u, Gr_j, j) for each user u ∈ G. """
        sat = {}
        for u in group:
            num = float(M.loc[gr_j, u].sum()) if gr_j else 0.0
            den = float(A[u].head(len(gr_j)).sum()) if u in A and not A[u].empty else 0.0
            sat[u] = num / den
        return pd.Series(sat, dtype=float)


    def run_sdaa_session(self, group: list, sim, rounds: int = 5, gamma: float = 0.5) -> list:
        """ Execute several iterations of the Satisfaction-Weighted SDAA.

        - alpha_prev controls the trade-off between average and least-misery.
        - user_weights amplify the voice of under-satisfied users across rounds.
        - TAU is used as a stopping threshold on disagreement (alpha).
        """
        alpha_prev = 0.0
        chosen = set()
        results = []

        # initialize all users with equal weight
        user_weights = {u: 1.0 for u in group}

        for j in range(1, rounds + 1):
            print(f"\n--- Iteration {j} ---")

            # 1) Individual top-k lists A(u, j), excluding already chosen items
            A = {
                u: self.user_recommendations(
                    u, self.ratings,
                    sim,
                    top_k=self.topn,
                    exclude=(chosen if USE_EXCLUSION else None)

                )
                for u in group
            }

            # 2) Gl = union of all A(u, j)
            Gl = sorted(set().union(*[s.index.tolist() for s in A.values() if not s.empty]))
            if not Gl:
                print(f"[Iter {j}] GL empty – stopping.")
                break

            # 3) Build prediction matrix M only for Gl
            M = self._initialize_group(group, self.ratings, sim, k=K, movies=Gl)
            if M.empty:
                print(f"[Iter {j}] M empty – stopping.")
                break

            k_j = min(self.topn, len(M))

            # 4) Compute SW-SDAA scores with satisfaction-based weights
            scores = self._sdaa_scores(M, alpha_prev, user_weights=user_weights)
            gr_j = scores.nlargest(k_j).index.tolist()

            # 5) Compute user satisfaction
            user_sat = self._user_satisfaction(M, group, gr_j, A)
            alpha_raw = float(user_sat.max() - user_sat.min())
            # keep alpha in [0, 1]
            alpha_j = max(0.0, min(1.0, alpha_raw))

            # 6) Update user weights: boost under-satisfied users
            mean_sat = float(user_sat.mean())
            for u in group:
                # positive if user is below mean satisfaction
                delta = mean_sat - float(user_sat.get(u, mean_sat))
                # multiplicative update; gamma controls strength
                user_weights[u] *= (1.0 + gamma * delta)

            # 7) Store iteration result
            results.append({
                "iteration": j,
                "alpha_used": alpha_prev,
                "alpha_next": alpha_j,
                "user_sat": user_sat.to_dict(),
                "user_weights": user_weights.copy(),
                "top_items": [
                    (int(mid), self.get_movie_name(int(mid)), float(scores.loc[mid]))
                    for mid in gr_j
                ],
            })

            print("User satisfactions:")
            for u, s in user_sat.items():
                print(f"  u={u}: {s:.3f}")
            print("User weights (next iter):")
            for u, w in user_weights.items():
                print(f"  u={u}: {w:.3f}")
            print(f"New α = {alpha_j:.3f}")
            print("Top recommendations:")
            for mid in gr_j:
                print(f"  - {self.get_movie_name(mid)} ({scores.loc[mid]:.2f})")

            # 8) Add chosen items to exclusion set
            chosen.update(int(x) for x in gr_j)
            alpha_prev = alpha_j

            # 9) Stopping rule based on TAU
            if alpha_prev < TAU:
                print(f"[Iter {j}] Disagreement α={alpha_prev:.3f} below TAU={TAU:.3f} – stopping.")
                break

        return results

    
    # --- INTERACTIVE CLI ---

    def cli(self):
        print("\n=== Recommender CLI ===")
        print("p = toggle method (Pearson <-> Spearman+shrink)")
        print("1 = predict (userId + movieId) using SELECTED method")
        print("2 = group recommendations (Average) from seed user using SELECTED method")
        print("3 = group recommendations (Least Misery) from seed user using SELECTED method")
        print("4 = run SDAA session from seed user using SELECTED method")
        print("q = quit\n")

        def method_label():
            return "PEARSON" if self.current_method == 'pearson' else "SPEARMAN+shrink"

        while True:
            cmd = input(f"[method={method_label()}] Select [p/1/2/3/4/q]: ").strip().lower()
            if cmd == 'q':
                print("Bye.")
                break

            elif cmd == 'p':
                self.current_method = 'pearson' if self.current_method != 'pearson' else 'spearman'
                print(f"-> Method switched to {method_label()}\n")

            elif cmd == '1':
                try:
                    u = int(input(" userId: ").strip())
                    m = int(input(" movieId: ").strip())
                    sim = self._active_sim()
                    score = self.predict_score_cached(u, m, sim, self.ratings, k=K)
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
                        group = self.topk_neighbors_group(seed_user=seed, sim=sim, k=GROUP_SIZE-1)
                    print(f" Group users (seed={seed}, method={method_label()}): {group}")
                    print()
                    recs = self.group_recs(self.ratings, sim, group, method="average", k=self.topn)
                    if recs.empty:
                        print(" No candidates found.\n")
                        continue
                    print(f" Top-{self.topn} (AVERAGE):")
                    for mid, sc in recs.items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    print(f" LEAST DISAGREEMENT (TOP-{self.topn}):")
                    for mid, sc in self.rank_by_lowest_disagreement(recs.index.tolist()).items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    consensus_recs = self.group_recs_consensus()
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
                        group = self.topk_neighbors_group(seed_user=seed, sim=sim, k=GROUP_SIZE-1)
                    print(f" Group users (seed={seed}, method={method_label()}): {group}")
                    print()
                    recs = self.group_recs(self.ratings, sim, group, method="least_misery", k=self.topn)
                    if recs.empty:
                        print(" No candidates found.\n")
                        continue
                    print(f" Top-{self.topn} (LEAST MISERY):")
                    for mid, sc in recs.items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    print(f" LEAST DISAGREEMENT (TOP-{self.topn}):")
                    for mid, sc in self.rank_by_lowest_disagreement(recs.index.tolist()).items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()
                    consensus_recs = self.group_recs_consensus()
                    if consensus_recs.empty:
                        print(" No candidates found for disagreement.\n")
                        continue
                    print(" CONSENSUS (ALL MOVIES):")
                    for mid, sc in consensus_recs.items():
                        print(f"  - {self.get_movie_name(mid)} (ID {mid}): {sc:.2f}")
                    print()

                except Exception as e:
                    print(f"Error: {e}\n")

            elif cmd == '4':
                seed = int(input(" seed userId: (0 for random group): ").strip())
                sim = self._active_sim()
                if seed == 0:
                    group = np.random.choice(a=self.ratings.index.tolist(), size=GROUP_SIZE, replace=False).tolist()
                else:
                    group = self.topk_neighbors_group(seed_user=seed, sim=sim, k=GROUP_SIZE-1)
                rounds = int(input(" number of rounds (e.g. 5): ").strip())
                print(f" Group users (seed={seed}, method={method_label()}, rounds={rounds}): {group}")
                print()
                results = self.run_sdaa_session(group=group, sim=sim, rounds=rounds)

            else:
                print("Unknown command.\n")

if __name__ == "__main__":
    recommender = RecommenderSystem()
    print(f"Dataset rows: {recommender.ratings_df.shape[0]}, columns: {recommender.ratings_df.shape[1]}")
    print(f"Users: {len(recommender.ratings.index)}, Movies: {len(recommender.ratings.columns)}\n")
    recommender.cli()