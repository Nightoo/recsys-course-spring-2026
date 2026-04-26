import json
import random
import numpy as np
from collections import defaultdict
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.fallback = fallback_recommender
        self.num_factors = 50
        self.lr = 0.01
        self.reg = 0.01
        self.epochs = 30
        self.user_vecs = {}
        self.track_vecs = {}
        self._bpr_train()

    def _bpr_train(self):
        keys = self.listen_history_redis.keys("user:*:listens")
        user_pos = defaultdict(set)
        all_tracks = set()
        for key in keys:
            user_id = int(key.decode().split(':')[1])
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                entry = json.loads(raw)
                track_id = int(entry["track"])
                user_pos[user_id].add(track_id)
                all_tracks.add(track_id)

        if not user_pos or not all_tracks:
            return

        for u in user_pos:
            self.user_vecs[u] = np.random.normal(0, 0.1, self.num_factors)
        for t in all_tracks:
            self.track_vecs[t] = np.random.normal(0, 0.1, self.num_factors)

        track_list = list(all_tracks)

        for epoch in range(self.epochs):
            users = list(user_pos.keys())
            random.shuffle(users)
            for u in users:
                pos_set = user_pos[u]
                if len(pos_set) == 0:
                    continue
                pos = random.sample(pos_set, 1)[0]
                neg = random.choice(track_list)
                while neg in pos_set:
                    neg = random.choice(track_list)
                x_ui = np.dot(self.user_vecs[u], self.track_vecs[pos])
                x_uj = np.dot(self.user_vecs[u], self.track_vecs[neg])
                sig = 1.0 / (1.0 + np.exp(x_uj - x_ui))
                grad_u = (self.track_vecs[pos] - self.track_vecs[neg]) * sig - self.reg * self.user_vecs[u]
                grad_pos = self.user_vecs[u] * sig - self.reg * self.track_vecs[pos]
                grad_neg = -self.user_vecs[u] * sig - self.reg * self.track_vecs[neg]
                self.user_vecs[u] += self.lr * grad_u
                self.track_vecs[pos] += self.lr * grad_pos
                self.track_vecs[neg] += self.lr * grad_neg

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
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
            if track_id in seen:
                continue
            score = np.dot(u_vec, t_vec)
            if score > best_score:
                best_score = score
                best_track = track_id

        if best_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        return best_track
