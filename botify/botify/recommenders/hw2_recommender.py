import json
from collections import defaultdict, Counter
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.fallback = fallback_recommender
        self.transitions = defaultdict(float)
        self.top_transitions = {}
        self._build_transitions()

    def _build_transitions(self):
        keys = self.listen_history_redis.keys("user:*:listens")
        for key in keys:
            entries = []
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                entry = json.loads(raw)
                entries.append((float(entry["time"]), int(entry["track"])))
            entries.sort(key=lambda x: x[0])
            for i in range(len(entries) - 1):
                prev_time, prev_track = entries[i]
                _, next_track = entries[i+1]
                self.transitions[(prev_track, next_track)] += prev_time

        prev_map = defaultdict(list)
        for (prev, nxt), weight in self.transitions.items():
            prev_map[prev].append((weight, nxt))
        for prev, lst in prev_map.items():
            lst.sort(reverse=True)
            self.top_transitions[prev] = lst[:50]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        seen = set()
        for raw in raw_entries:
            entry = json.loads(raw)
            seen.add(int(entry["track"]))

        if prev_track not in self.top_transitions:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        for weight, cand in self.top_transitions[prev_track]:
            if cand not in seen:
                return cand

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
