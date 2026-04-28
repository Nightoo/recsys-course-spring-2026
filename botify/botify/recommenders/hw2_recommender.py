import pickle
import numpy as np

class Solution:
    def __init__(self, listen_history_redis, random_recommender, model_path="data/model.pkl", top_candidates=100):
        self.listen_history_redis = listen_history_redis
        self.random_recommender = random_recommender
        self.top_candidates = top_candidates
        self.model = None
        self.track_freq = {}
        self.track_avg_time = {}
        self.track_emb = {}
        self.sorted_tracks = []
        try:
            with open(model_path, "rb") as f:
                self.model, self.track_freq, self.track_avg_time, self.track_emb = pickle.load(f)
            self.sorted_tracks = sorted(self.track_freq.items(), key=lambda x: x[1], reverse=True)
        except:
            pass

    def _get_candidates(self, last_track):
        candidates = [track for track, _ in self.sorted_tracks[:self.top_candidates]]
        if last_track in candidates:
            candidates.remove(last_track)
        return candidates

    def _compute_features(self, prev_track, cand_track, pos, session_len):
        emb_prev = self.track_emb.get(prev_track, np.zeros(128))
        emb_cand = self.track_emb.get(cand_track, np.zeros(128))
        cos = np.dot(emb_prev, emb_cand) / (np.linalg.norm(emb_prev)*np.linalg.norm(emb_cand)+1e-9)
        return [
            self.track_freq.get(prev_track, 0),
            self.track_freq.get(cand_track, 0),
            self.track_avg_time.get(prev_track, 0.0),
            self.track_avg_time.get(cand_track, 0.0),
            pos,
            session_len,
            cos
        ]

    def recommend_next(self, user, last_track, last_time):
        if self.model is None:
            return self.random_recommender.recommend_next(user, last_track, last_time)
        try:
            history_len = self.listen_history_redis.llen(f"user:{user}:listens")
            pos = history_len
            session_len = history_len + 1
            candidates = self._get_candidates(last_track)
            if not candidates:
                return self.random_recommender.recommend_next(user, last_track, last_time)
            best_score = -1
            best_track = None
            for cand in candidates:
                feats = self._compute_features(last_track, cand, pos, session_len)
                pred = self.model.predict([feats])[0]
                if pred > best_score:
                    best_score = pred
                    best_track = cand
            return best_track if best_track is not None else self.random_recommender.recommend_next(user, last_track, last_time)
        except:
            return self.random_recommender.recommend_next(user, last_track, last_time)
