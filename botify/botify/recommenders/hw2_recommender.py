import json
import numpy as np
from collections import defaultdict
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.fallback = fallback_recommender
        self.similar_users = {}
        self.user_tracks = {}
        self._build_user_similarity()

    def _build_user_similarity(self):
        keys = self.listen_history_redis.keys("user:*:listens")
        for key in keys:
            user_id = int(key.decode().split(':')[1])
            tracks = set()
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                entry = json.loads(raw)
                tracks.add(int(entry["track"]))
            if tracks:
                self.user_tracks[user_id] = tracks

        if len(self.user_tracks) < 2:
            return

        user_list = list(self.user_tracks.keys())
        user_idx = {u: i for i, u in enumerate(user_list)}
        num_users = len(user_list)
        all_tracks = set()
        for tracks in self.user_tracks.values():
            all_tracks.update(tracks)
        track_to_idx = {t: i for i, t in enumerate(all_tracks)}
        num_tracks = len(all_tracks)
        matrix = np.zeros((num_users, num_tracks), dtype=bool)
        for u, tracks in self.user_tracks.items():
            u_i = user_idx[u]
            for t in tracks:
                if t in track_to_idx:
                    matrix[u_i, track_to_idx[t]] = True
        norm = np.sqrt(matrix.sum(axis=1, keepdims=True))
        norm[norm == 0] = 1
        matrix_norm = matrix / norm
        sim_matrix = matrix_norm @ matrix_norm.T  # (num_users, num_users)
        for i, u in enumerate(user_list):
            sims = [(sim_matrix[i, j], user_list[j]) for j in range(num_users) if j != i]
            sims.sort(reverse=True)
            self.similar_users[u] = sims[:50]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        key = f"user:{user}:listens"
        history = set()
        for raw in self.listen_history_redis.lrange(key, 0, -1):
            entry = json.loads(raw)
            history.add(int(entry["track"]))

        if user not in self.similar_users:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        candidate_scores = defaultdict(float)
        for sim, other_user in self.similar_users[user]:
            other_tracks = self.user_tracks.get(other_user, set())
            for track in other_tracks:
                if track not in history:
                    candidate_scores[track] += sim

        if not candidate_scores:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        best_track = max(candidate_scores, key=candidate_scores.get)
        return best_track
