import json
import random
import numpy as np
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.tracks_redis = tracks_redis
        self.catalog = catalog
        self.fallback = fallback_recommender

        self.num_factors = 30
        self.lr = 0.05
        self.reg = 0.02
        self.is_trained = False

        self.user_vecs = {}
        self.track_vecs = {}

    def _init_training(self):
        keys = self.listen_history_redis.keys("user:*:listens")
        for key in keys:
            user_id = int(key.decode().split(':')[1])
            raw_entries = self.listen_history_redis.lrange(key, 0, -1)
            for raw in raw_entries:
                entry = json.loads(raw)
                track_id = int(entry["track"])
                time_val = float(entry["time"])
                self._sgd_update(user_id, track_id, time_val)
        self.is_trained = True

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
        if not self.is_trained:
            self._init_training()

        self._sgd_update(user, prev_track, prev_track_time)

        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        seen = set()
        for raw in raw_entries:
            entry = json.loads(raw)
            seen.add(int(entry["track"]))

        if user not in self.user_vecs:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        u_vec = self.user_vecs[user]
        candidates = []
        for track_id in self.catalog.tracks.keys():
            if track_id in seen:
                continue
            if track_id not in self.track_vecs:
                self._ensure_vector(track_id, self.track_vecs)
            score = np.dot(u_vec, self.track_vecs[track_id])
            candidates.append((score, track_id))

        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidates.sort(reverse=True)
        return candidates[0][1]
