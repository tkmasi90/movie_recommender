class SessionManager:
    def __init__(self, group_users, movies_df, gamma=0.95):
        self.group_users = group_users
        self.movies_df = movies_df
        self.history = []
        self.satisfaction = {u: 0.0 for u in group_users}
        self.gamma = gamma
        self.t = 0

    def update(self, movie_id: int, predicted_ratings: dict[int,float]):
        self.history.append(movie_id)
        self.t += 1
        disc = self.gamma ** self.t
        for u, r in predicted_ratings.items():
            self.satisfaction[u] += disc * float(r)