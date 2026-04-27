import json
from collections import defaultdict, Counter
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, artists_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.fallback = fallback_recommender

        self.transitions = defaultdict(Counter)
        self.popularity = Counter()
        self.artist_tracks = defaultdict(list)

        self._build_artist_map()
        self._build_transitions()

    def _build_artist_map(self):
        for track_id, track_data in self.catalog.tracks.items():
            artist = track_data.get('artist') or track_data.get('artist_id')
            if artist:
                self.artist_tracks[artist].append(track_id)

    def _build_transitions(self):
        keys = self.listen_history_redis.keys("user:*:listens")
        for key in keys:
            entries = []
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                e = json.loads(raw)
                track_id = int(e["track"])
                listen_time = float(e["time"])
                entries.append((listen_time, track_id))
                self.popularity[track_id] += 1
            entries.sort(key=lambda x: x[0])
            for i in range(len(entries) - 1):
                prev = entries[i][1]
                nxt = entries[i+1][1]
                self.transitions[prev][nxt] += 1

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        key = f"user:{user}:listens"
        seen = set()
        for raw in self.listen_history_redis.lrange(key, 0, -1):
            seen.add(int(json.loads(raw)["track"]))

        if prev_track_time > 30:
            artist = self.artists_redis.get(str(prev_track))
            if artist:
                if isinstance(artist, bytes):
                    artist = artist.decode()
                for track in self.artist_tracks.get(artist, []):
                    if track not in seen:
                        return track

        if prev_track in self.transitions:
            for nxt, _ in self.transitions[prev_track].most_common(50):
                if nxt not in seen:
                    return nxt

        for track, _ in self.popularity.most_common():
            if track not in seen:
                return track

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
