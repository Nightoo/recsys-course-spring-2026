import json
import os
import numpy as np
import pandas as pd
from .recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, catalog, fallback_recommender,
                 num_factors=30, lr=0.05, reg=0.02, epochs=20,
                 redis_min_records=1000, refit_threshold=5000):
        self.redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback_recommender
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.redis_min_records = redis_min_records
        self.refit_threshold = refit_threshold

        self.user_to_idx = {}
        self.track_to_idx = {}
        self.idx_to_track = {}
        self.user_vecs = None
        self.track_vecs = None
        self.is_fitted = False
        self.last_train_count = 0

    def fit(self, csv_path='train.csv'):
        redis_df = self._load_redis()
        if len(redis_df) >= self.redis_min_records:
            df = redis_df
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

        self.last_train_count = len(redis_df) if len(redis_df) >= self.redis_min_records else len(df)

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
                loss += err*err

        self.is_fitted = True

    def _get_total_redis_records(self):
        keys = self.redis.keys("user:*:listens")
        total = 0
        for key in keys:
            total += self.redis.llen(key)
        return total

    def _should_refit(self):
        if not self.is_fitted:
            return False
        current_total = self._get_total_redis_records()
        return (current_total - self.last_train_count) >= self.refit_threshold

    def _load_redis(self):
        keys = self.redis.keys("user:*:listens")
        if not keys:
            return pd.DataFrame(columns=['user', 'track', 'time'])
        data = []
        for key in keys:
            uid = int(key.split(b':')[1])
            entries = self.redis.lrange(key, 0, -1)
            for raw in entries:
                if isinstance(raw, bytes):
                    raw = raw.decode()
                entry = json.loads(raw)
                data.append([uid, entry['track'], entry['time']])
        df = pd.DataFrame(data, columns=['user', 'track', 'time'])
        return df

    def recommend_next(self, user, prev_track, prev_track_time):
        if self.is_fitted and self._should_refit():
            self.fit()

        if not self.is_fitted:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        history = self._load_user_history(user)
        seen = {track for track, _ in history}
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

    def _load_user_history(self, user):
        key = f"user:{user}:listens"
        raw = self.redis.lrange(key, 0, -1)
        res = []
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode()
            data = json.loads(r)
            res.append((data['track'], data['time']))
        return res
