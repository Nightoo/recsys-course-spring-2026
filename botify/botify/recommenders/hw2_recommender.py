import json
import os
import numpy as np
import pandas as pd
from .recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, catalog, fallback_recommender,
                 num_factors=30, lr=0.05, reg=0.02, epochs=20):
        self.redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback_recommender
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.is_fitted = False
        self.user_to_idx = {}
        self.track_to_idx = {}
        self.idx_to_track = {}
        self.user_vecs = None
        self.track_vecs = None

    def fit(self, csv_path='train.csv'):
        try:
            if not os.path.exists(csv_path):
                return
            df = pd.read_csv(csv_path)
            if len(df) < 5:
                return

            users = df['user'].unique()
            tracks = df['track'].unique()
            self.user_to_idx = {u: i for i, u in enumerate(users)}
            self.track_to_idx = {t: i for i, t in enumerate(tracks)}
            self.idx_to_track = {i: t for t, i in self.track_to_idx.items()}

            ratings = []
            for _, row in df.iterrows():
                u = self.user_to_idx[row['user']]
                t = self.track_to_idx[row['track']]
                w = 1.0 + np.log1p(row['time'])
                ratings.append((u, t, w))

            self.user_vecs = np.random.normal(0, 0.1, (len(users), self.num_factors))
            self.track_vecs = np.random.normal(0, 0.1, (len(tracks), self.num_factors))

            for epoch in range(self.epochs):
                np.random.shuffle(ratings)
                loss = 0.0
                for u, t, w in ratings:
                    pred = np.dot(self.user_vecs[u], self.track_vecs[t])
                    err = w - pred
                    grad_u = -2 * err * self.track_vecs[t] + 2 * self.reg * self.user_vecs[u]
                    grad_t = -2 * err * self.user_vecs[u] + 2 * self.reg * self.track_vecs[t]
                    self.user_vecs[u] -= self.lr * grad_u
                    self.track_vecs[t] -= self.lr * grad_t
                    loss += err * err

            self.is_fitted = True
        except Exception as e:
            print(f"{e}")

    def recommend_next(self, user, prev_track, prev_track_time):
        if not self.is_fitted:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        key = f"user:{user}:listens"
        raw = self.redis.lrange(key, 0, -1)
        seen = set()
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode()
            data = json.loads(r)
            seen.add(data['track'])
        seen.add(prev_track)

        if user not in self.user_to_idx:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        u = self.user_to_idx[user]
        best_track = None
        best_score = -np.inf
        for track, t_idx in self.track_to_idx.items():
            if track in seen:
                continue
            score = np.dot(self.user_vecs[u], self.track_vecs[t_idx])
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        return best_track
