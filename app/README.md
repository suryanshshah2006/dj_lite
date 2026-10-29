# DJ-Lite web app

Local web app for the DJ-Lite recommender: search tracks, see what mixes into them,
and build energy-arc mixes. Spotify-inspired layout, own brand.

```
app/
  backend/    FastAPI (imports the djlite package)      -> http://localhost:8000
  frontend/   Next.js App Router + TypeScript + Tailwind -> http://localhost:3000
```

## Frontend

```powershell
cd app/frontend
npm install                  # first time only
npm run build; npm start     # production build, http://localhost:3000 (recommended)
npm run dev                  # dev server with hot reload
```

By default the frontend calls the **live backend** at `http://localhost:8000`
(override with `NEXT_PUBLIC_API_BASE`). For offline demos without the backend, switch to the
**mock API** (30 real tracks in `lib/mock/tracks.json`, scoring mirrored in `lib/mock/engine.ts`):

```powershell
# app/frontend/.env.local
NEXT_PUBLIC_API_MODE=mock
```

`NEXT_PUBLIC_*` values are baked in at build time: rebuild (or restart `npm run dev`) after
changing `.env.local`.

## Backend

Run from the project root with the project venv active (`.venv\Scripts\Activate.ps1`):

```powershell
uv pip install -r app/backend/requirements.txt   # first time only
python -m app.backend.build_search_index          # first time + whenever track_features changes (~75 s)
uvicorn app.backend.main:app --port 8000
```

At startup it loads the track features, style vectors and FAISS index once: about 25 s and
~2.3 GB RAM. `app/backend/__init__.py` pins OpenBLAS/OpenMP to one thread; with the default of
one thread per core the bundled BLAS copies and faiss's workers commit an extra ~3 GB for no
measurable speed-up on single-query search.

Search uses a SQLite FTS5 index over title + artists (`app/backend/cache/search.sqlite`,
~255 MB, gitignored): exact title matches first, then title prefixes, then artist matches,
then any-field matches. Within a tier, results are ordered by the primary artist's
prominence, precomputed at build time (`search.prominence` in `settings.yaml`):

- `co_artists` (default): distinct artists sharing an album with them. Originals appear on
  compilations and soundtracks, cover/tribute acts mostly on single-artist albums, so
  "bohemian rhapsody" returns Queen first and "shape of you" Ed Sheeran.
- `tracks`: number of tracks by the artist. Favours prolific cover acts (Vitamin String
  Quartet, 1,474 tracks, comes first for both queries above).

Then the duplicate-release popularity proxy, then shorter titles. Specific queries take
~1–6 ms; one- or two-letter prefixes ~50–70 ms (prominence ordering looks at the first
`tier_window` matches). Without the index the backend falls back to a slow in-memory scan.

Tests use a small synthetic fixture and never load the full dataset:

```powershell
python -m pytest app/backend/tests -q
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/search?q=` | Top 20 tracks by name or artist |
| GET | `/api/featured` | Curated tracks for the Home page (`featured:` in `settings.yaml`) |
| GET | `/api/track/{id}` | Track details (BPM, Camelot, energy, valence, …) |
| POST | `/api/next` | `{seed_id, n, alpha}` → next tracks with score breakdown |
| POST | `/api/mix` | `{seed_ids, arc, alpha, length, generator?}` → arc-following mix; `beam_v1` (default) or `greedy_v0`, with `runtime_ms` |
| GET | `/api/cover/{id}` | Cover URL via Spotify oEmbed, SQLite-cached, ~5 req/s |
| GET | `/api/eval` | Evaluation table + figure once `reports/eval_v2.*` exist |

## Album art and playback

- Covers: `https://open.spotify.com/oembed?url=https://open.spotify.com/track/{id}` →
  `thumbnail_url`. No API keys. Only URLs are stored (never images), only for tracks being
  displayed (never bulk), with failures cached so they are not retried endlessly.
- No cover → a gradient tile with the track's initials, coloured by its Camelot key.
- Playback: the official embed iframe `open.spotify.com/embed/track/{id}` in the bottom bar.

Artwork & playback via Spotify. Audio features from the Spotify 1.2M+ Songs dataset (Kaggle).
