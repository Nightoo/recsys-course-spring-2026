import json
from collections import defaultdict
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback_recommender
        self.top_artist_tracks = defaultdict(list)
        self.user_favorite_artists = defaultdict(lambda: defaultdict(float))
        self._load_data()

    def _load_data(self):
        track_to_artist = {}
        for track_id, track_info in self.catalog.tracks.items():
            artist = None
            if isinstance(track_info, dict):
                artist = track_info.get('artist')
            else:
                artist = getattr(track_info, 'artist', None)
            if artist:
                track_to_artist[track_id] = artist
                self.top_artist_tracks[artist].append(track_id)

        keys = self.listen_history_redis.keys("user:*:listens")
        for key in keys:
            user_id = int(key.decode().split(':')[1])
            entries = []
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                entry = json.loads(raw)
                track = int(entry["track"])
                time_val = float(entry["time"])
                entries.append((time_val, track))
            entries.sort(key=lambda x: x[0])
            for time_val, track in entries:
                artist = track_to_artist.get(track)
                if artist:
                    self.user_favorite_artists[user_id][artist] += time_val

        for artist in self.top_artist_tracks:
            pass

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        try:
            key = f"user:{user}:listens"
            seen = set()
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                entry = json.loads(raw)
                seen.add(int(entry["track"]))

            artist_prefs = self.user_favorite_artists.get(user, {})
            if artist_prefs:
                sorted_artists = sorted(artist_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
                for artist, _ in sorted_artists:
                    for track in self.top_artist_tracks.get(artist, []):
                        if track not in seen:
                            return track

            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        except Exception:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
