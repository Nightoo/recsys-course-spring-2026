import json
import random
import numpy as np
from collections import defaultdict
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.fallback = fallback_recommender
        self.track_similarity = {}
        self.track_history = defaultdict(set)
        self._update_from_history()

    def _update_from_history(self):
        keys = self.listen_history_redis.keys("user:*:listens")
        user_tracks = {}  # user_id -> set of track_ids
        for key in keys:
            user_id = int(key.decode().split(':')[1])
            tracks = set()
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                entry = json.loads(raw)
                track_id = int(entry["track"])
                tracks.add(track_id)
                self.track_history[track_id].add(user_id)
            if tracks:
                user_tracks[user_id] = tracks

        track_list = list(self.track_history.keys())
        if len(track_list) > 1000:
            pop = [(len(self.track_history[t]), t) for t in track_list]
            pop.sort(reverse=True)
            track_list = [t for _, t in pop[:1000]]

        track_idx = {t: i for i, t in enumerate(track_list)}
        num_users = len(user_tracks)
        num_tracks = len(track_list)
        if num_tracks == 0:
            return
        matrix = np.zeros((num_users, num_tracks), dtype=bool)
        user_idx = {u: i for i, u in enumerate(user_tracks.keys())}
        for u, tracks in user_tracks.items():
            u_i = user_idx[u]
            for t in tracks:
                if t in track_idx:
                    matrix[u_i, track_idx[t]] = True

        norm = np.sqrt(matrix.sum(axis=1, keepdims=True))
        norm[norm == 0] = 1
        matrix_norm = matrix / norm
        sim = matrix_norm.T @ matrix_norm  # (num_tracks, num_tracks)
        for i, t in enumerate(track_list):
            scores = [(sim[i, j], track_list[j]) for j in range(num_tracks) if j != i and sim[i, j] > 0]
            scores.sort(reverse=True)
            self.track_similarity[t] = scores[:20]  # топ-20

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        key = f"user:{user}:listens"
        history_tracks = set()
        for raw in self.listen_history_redis.lrange(key, 0, -1):
            entry = json.loads(raw)
            history_tracks.add(int(entry["track"]))

        if prev_track not in self.track_similarity:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        for score, cand in self.track_similarity[prev_track]:
            if cand not in history_tracks:
                return cand

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
