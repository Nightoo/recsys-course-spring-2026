import json
import random
import numpy as np
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.fallback = fallback_recommender
        self.num_factors = 30
        self.lr = 0.05
        self.reg = 0.02
        self.user_vecs = {}
        self.track_vecs = {}

    def _ensure_vector(self, entity_id, vec_dict):
        if entity_id not in vec_dict:
            vec_dict[entity_id] = np.random.normal(0, 0.1, self.num_factors)

    def _sgd_update(self, user_id, track_id, time_val):
        self._ensure_vector(user_id, self.user_vecs)
        self._ensure_vector(track_id, self.track_vecs)
        u = self.user_vecs[user_id]
        t = self.track_vecs[track_id]
        pred = np.dot(u, t)
        err = time_val - pred
        grad_u = -2 * err * t + 2 * self.reg * u
        grad_t = -2 * err * u + 2 * self.reg * t
        self.user_vecs[user_id] = u - self.lr * grad_u
        self.track_vecs[track_id] = t - self.lr * grad_t

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        self._sgd_update(user, prev_track, prev_track_time)

        key = f"user:{user}:listens"
        seen = set()
        for raw in self.listen_history_redis.lrange(key, 0, -1):
            entry = json.loads(raw)
            seen.add(int(entry["track"]))

        if user not in self.user_vecs:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        u_vec = self.user_vecs[user]
        best_track = None
        best_score = -float('inf')
        for track_id, t_vec in self.track_vecs.items():
            if track_id not in seen:
                score = np.dot(u_vec, t_vec)
                if score > best_score:
                    best_score = score
                    best_track = track_id

        if best_track is not None:
            return best_track
        return self.fallback.recommend_next(user, prev_track, prev_track_time)
