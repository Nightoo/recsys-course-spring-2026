import json
from collections import defaultdict, Counter
from botify.recommenders.recommender import Recommender

class Solution(Recommender):
    def __init__(self, listen_history_redis, tracks_redis, catalog, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback_recommender

        self.user_artist_pref = defaultdict(float)
        self.artist_top_tracks = defaultdict(list)
        self.global_popular = []
        self.transitions = defaultdict(Counter)

        self._build_model()

    def _build_model(self):
        track_artist = {}
        for track_id, track in self.catalog.tracks.items():
            artist = track.get('artist') if isinstance(track, dict) else getattr(track, 'artist', None)
            if artist:
                track_artist[track_id] = artist

        keys = self.listen_history_redis.keys("user:*:listens")
        track_popularity = Counter()
        for key in keys:
            user_id = int(key.decode().split(':')[1])
            entries = []
            for raw in self.listen_history_redis.lrange(key, 0, -1):
                entry = json.loads(raw)
                track = int(entry["track"])
                time_ = float(entry["time"])
                entries.append((time_, track))
                track_popularity[track] += 1

                artist = track_artist.get(track)
                if artist:
                    self.user_artist_pref[(user_id, artist)] += time_

            entries.sort(key=lambda x: x[0])
            for i in range(len(entries)-1):
                prev = entries[i][1]
                nxt = entries[i+1][1]
                self.transitions[prev][nxt] += 1

        self.global_popular = [t for t, _ in track_popularity.most_common(100)]

        artist_track_pop = defaultdict(Counter)
        for track, cnt in track_popularity.items():
            artist = track_artist.get(track)
            if artist:
                artist_track_pop[artist][track] = cnt
        for artist, counter in artist_track_pop.items():
            self.artist_top_tracks[artist] = [t for t, _ in counter.most_common(20)]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        key = f"user:{user}:listens"
        seen = set()
        for raw in self.listen_history_redis.lrange(key, 0, -1):
            entry = json.loads(raw)
            seen.add(int(entry["track"]))

        user_artists = [(artist, time) for (u, artist), time in self.user_artist_pref.items() if u == user]
        user_artists.sort(key=lambda x: x[1], reverse=True)
        for artist, _ in user_artists[:5]:
            for track in self.artist_top_tracks.get(artist, []):
                if track not in seen:
                    return track

        if prev_track in self.transitions:
            for nxt, _ in self.transitions[prev_track].most_common(30):
                if nxt not in seen:
                    return nxt

        for track in self.global_popular:
            if track not in seen:
                return track

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
