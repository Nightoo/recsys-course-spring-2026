class Solution:
    def __init__(self, listen_history_redis, random_recommender):
        self.listen_history_redis = listen_history_redis
        self.random_recommender = random_recommender

    def recommend_next(self, user, last_track, last_time):
        return 1000
