import pickle
import numpy as np

class Solution:
    def __init__(self, listen_history_redis, random_recommender, model_path="data/model.pkl", top_candidates=100):
        self.listen_history_redis = listen_history_redis
        self.random_recommender = random_recommender
        self.top_candidates = top_candidates
        with open(model_path, "rb") as f:
            self.model, self.track_freq, self.track_avg_time, self.embeddings = pickle.load(f)
        self.all_tracks = list(self.track_freq.keys())

    def _get_candidates(self, last_track, history_len):
        sorted_tracks = sorted(self.track_freq.items(), key=lambda x: x[1], reverse=True)
        candidates = [track for track, _ in sorted_tracks[:self.top_candidates]]
        if last_track in candidates:
            candidates.remove(last_track)
        return candidates

    def _compute_features(self, prev_track, cand_track, pos, session_len):
        emb_prev = self.embeddings.get(prev_track, np.zeros(64))
        emb_cand = self.embeddings.get(cand_track, np.zeros(64))
        cos_sim = np.dot(emb_prev, emb_cand) / (np.linalg.norm(emb_prev)*np.linalg.norm(emb_cand)+1e-9)
        return [
            self.track_freq.get(prev_track, 0),
            self.track_freq.get(cand_track, 0),
            self.track_avg_time.get(prev_track, 0.0),
            self.track_avg_time.get(cand_track, 0.0),
            pos,
            session_len,
            cos_sim
        ]

    def recommend_next(self, user, last_track, last_time):
        history_key = f"user:{user}:listens"
        history_len = self.listen_history_redis.llen(history_key)
        pos = history_len
        candidates = self._get_candidates(last_track, history_len)
        if not candidates:
            return self.random_recommender.recommend_next(user, last_track, last_time)
        best_time = -1
        best_track = None
        for cand in candidates:
            feats = self._compute_features(last_track, cand, pos, history_len+1)
            pred = self.model.predict([feats])[0]
            if pred > best_time:
                best_time = pred
                best_track = cand
        if best_track is None:
            best_track = self.random_recommender.recommend_next(user, last_track, last_time)
        return best_track