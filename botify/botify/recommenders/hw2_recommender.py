from .recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, catalog, fallback_recommender):
        self.fallback = fallback_recommender

    def fit(self):
        pass

    def recommend_next(self, user, prev_track, prev_track_time):
        return self.fallback.recommend_next(user, prev_track, prev_track_time)
