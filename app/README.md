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
npm install          # first time only
npm run dev          # http://localhost:3000
```

The frontend talks to either a **mock API** (default; 30 real tracks in
`lib/mock/tracks.json`, scoring mirrored in `lib/mock/engine.ts`) or the **live backend**:

```powershell
# app/frontend/.env.local
NEXT_PUBLIC_API_MODE=live
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

Restart `npm run dev` after changing `.env.local`.

## Backend

Run from the project root with the project venv active (`.venv\Scripts\Activate.ps1`):

```powershell
uv pip install -r app/backend/requirements.txt   # first time only
uvicorn app.backend.main:app --port 8000
```

At startup it loads the track features, style vectors and FAISS index once (~1.5 GB RAM,
about a minute). Tests use a small synthetic fixture and never load the full dataset:

```powershell
python -m pytest app/backend/tests -q
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/search?q=` | Top 20 tracks by name or artist |
| GET | `/api/track/{id}` | Track details (BPM, Camelot, energy, valence, …) |
| POST | `/api/next` | `{seed_id, n, alpha}` → next tracks with score breakdown |
| POST | `/api/mix` | `{seed_ids, arc, alpha, length}` → arc-following mix (`generator: greedy_v0`) |
| GET | `/api/cover/{id}` | Cover URL via Spotify oEmbed, SQLite-cached, ~5 req/s |
| GET | `/api/eval` | Evaluation table + figure once `reports/eval_v2.*` exist |

## Album art and playback

- Covers: `https://open.spotify.com/oembed?url=https://open.spotify.com/track/{id}` →
  `thumbnail_url`. No API keys. Only URLs are stored (never images), only for tracks being
  displayed (never bulk), with failures cached so they are not retried endlessly.
- No cover → a gradient tile with the track's initials, coloured by its Camelot key.
- Playback: the official embed iframe `open.spotify.com/embed/track/{id}` in the bottom bar.

Artwork & playback via Spotify. Audio features from the Spotify 1.2M+ Songs dataset (Kaggle).
