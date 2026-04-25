import json
import random
from .recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, catalog, fallback_recommender):
        self.redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback_recommender
        self.all_tracks = list(self.catalog.tracks.keys())
        print(f"Solution initialized with {len(self.all_tracks)} tracks")

    def recommend_next(self, user, prev_track, prev_track_time):
        key = f"user:{user}:listens"
        raw = self.redis.lrange(key, 0, -1)
        seen = set()
        for r in raw:
            if isinstance(r, bytes):
                r = r.decode()
            data = json.loads(r)
            seen.add(int(data['track']))
        seen.add(prev_track)

        unseen = [t for t in self.all_tracks if t not in seen]
        if unseen:
            return random.choice(unseen)
        return self.fallback.recommend_next(user, prev_track, prev_track_time)
