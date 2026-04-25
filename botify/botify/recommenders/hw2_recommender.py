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

        self.user_vecs = None
        self.track_vecs = None
        self.user_to_idx = {}
        self.idx_to_user = []
        self.track_to_idx = {}
        self.idx_to_track = []
        self.call_count = 0
        self.RETRAIN_EVERY = 1000

    def _train(self):
        keys = self.listen_history_redis.keys("user:*:listens")
        if not keys:
            return

        user_history = {}
        all_tracks = set()
        for key in keys:
            user_id = int(key.decode().split(':')[1])
            raw_entries = self.listen_history_redis.lrange(key, 0, -1)
            history = []
            for raw in raw_entries:
                entry = json.loads(raw)
                track_id = int(entry["track"])
                time_val = float(entry["time"])
                history.append((track_id, time_val))
                all_tracks.add(track_id)
            if history:
                user_history[user_id] = history

        if not user_history or not all_tracks:
            return

        self.user_to_idx = {u: i for i, u in enumerate(user_history.keys())}
        self.idx_to_user = list(user_history.keys())
        self.track_to_idx = {t: i for i, t in enumerate(all_tracks)}
        self.idx_to_track = list(all_tracks)

        num_users = len(self.user_to_idx)
        num_tracks = len(self.track_to_idx)

        ratings = []
        for u, hist in user_history.items():
            u_idx = self.user_to_idx[u]
            for t, w in hist:
                if t in self.track_to_idx:
                    t_idx = self.track_to_idx[t]
                    ratings.append((u_idx, t_idx, w))

        num_factors = 30
        lr = 0.05
        reg = 0.02
        epochs = 25

        self.user_vecs = np.random.normal(0, 0.1, (num_users, num_factors))
        self.track_vecs = np.random.normal(0, 0.1, (num_tracks, num_factors))

        for _ in range(epochs):
            random.shuffle(ratings)
            for u, t, w in ratings:
                pred = np.dot(self.user_vecs[u], self.track_vecs[t])
                err = w - pred
                grad_u = -2 * err * self.track_vecs[t] + 2 * reg * self.user_vecs[u]
                grad_t = -2 * err * self.user_vecs[u] + 2 * reg * self.track_vecs[t]
                self.user_vecs[u] -= lr * grad_u
                self.track_vecs[t] -= lr * grad_t

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        self.call_count += 1
        if self.user_vecs is None or self.call_count % self.RETRAIN_EVERY == 0:
            self._train()

        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        seen = set()
        for raw in raw_entries:
            entry = json.loads(raw)
            seen.add(int(entry["track"]))

        if self.user_vecs is None or user not in self.user_to_idx:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        u_idx = self.user_to_idx[user]
        candidates = []
        for t_idx, track_id in enumerate(self.idx_to_track):
            if track_id not in seen:
                score = np.dot(self.user_vecs[u_idx], self.track_vecs[t_idx])
                candidates.append((score, track_id))

        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidates.sort(reverse=True)
        return candidates[0][1]
