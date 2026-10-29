"""Backend tests on a small synthetic fixture (never loads the real dataset)."""

from __future__ import annotations

import asyncio
import string
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backend.covers import CoverCache, FetchResult, RateLimiter
from app.backend.engine import Engine
from app.backend.main import create_app, load_settings
from djlite.features import add_camelot
from djlite.generator import load_generator_config
from djlite.retrieval import load_configs
from djlite.style import l2_normalize

AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo_folded"]
N = 240
ALPHABET = string.ascii_letters + string.digits


def _id(i: int) -> str:
    rng = np.random.default_rng(i)
    return "".join(rng.choice(list(ALPHABET), 22))


def fixture_tracks(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    words = ["night", "drive", "love", "fire", "ocean", "city", "dream", "gold", "echo", "run"]
    names = [f"{words[i % 10].title()} {words[(i // 10) % 10].title()} {i}" for i in range(N)]
    names[0] = "Ocean Drive"
    names[1] = "Ocean Drive (Remix)"
    df = pd.DataFrame({
        "id": [_id(i) for i in range(N)],
        "name": names,
        "artists": [[f"Artist {i % 60}"] + ([f"Guest {i % 7}"] if i % 9 == 0 else [])
                    for i in range(N)],
        "artist_ids": [[f"a{i % 60}"] + ([f"g{i % 7}"] if i % 9 == 0 else []) for i in range(N)],
        "album": [f"Album {i % 30}" for i in range(N)],
        "year": rng.integers(1975, 2021, N),
        "duration_ms": rng.integers(150_000, 360_000, N),
        "key": rng.integers(0, 12, N), "mode": rng.integers(0, 2, N),
        "tempo": rng.uniform(70, 175, N),
        "energy": rng.uniform(0.05, 0.98, N), "valence": rng.uniform(0, 1, N),
        "danceability": rng.uniform(0, 1, N), "acousticness": rng.uniform(0, 1, N),
        "instrumentalness": rng.uniform(0, 1, N), "speechiness": rng.uniform(0, 0.4, N),
        "liveness": rng.uniform(0, 0.6, N), "loudness": rng.uniform(-20, -2, N),
    })
    f = df["tempo"].to_numpy().copy()
    f = np.where(f < 80, f * 2, f)
    f = np.where(f >= 160, f / 2, f)
    df["tempo_folded"] = f
    for c in AUDIO:
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    df = add_camelot(df)
    style = l2_normalize(rng.normal(size=(N, 16)).astype(np.float32))
    style[:15] = 0  # some artists without a style vector
    return df, style


class FakeFetcher:
    def __init__(self, outcomes: dict | None = None):
        self.outcomes = outcomes or {}
        self.calls: list[tuple[str, float]] = []

    async def __call__(self, track_id: str) -> FetchResult:
        self.calls.append((track_id, time.monotonic()))
        await asyncio.sleep(0)
        return self.outcomes.get(track_id, FetchResult(f"https://img.example/{track_id}", "ok"))


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def engine(settings):
    _, rcfg, tcfg = load_configs()
    df, style = fixture_tracks()
    g = load_generator_config()
    # 240 random tracks: a tight |dBPM| cap can make a 10-track mix infeasible here; the cap
    # itself is covered by tests/test_generator.py
    g = {**g, "constraints": {**g["constraints"], "max_dbpm": None}}
    return Engine(df, style, rcfg, tcfg, settings, popularity=np.arange(N, dtype=float) % 5,
                  gcfg=g)


@pytest.fixture()
def client(engine, settings, tmp_path):
    fetcher = FakeFetcher({_id(5): FetchResult(None, "not_found"), _id(6): FetchResult(None, "error")})
    covers = CoverCache(tmp_path / "covers.sqlite", fetcher, rate_per_s=50)
    app = create_app(engine_factory=lambda s: engine, cover_cache=covers, settings=settings,
                     reports_root=tmp_path)
    with TestClient(app) as c:
        c.fetcher, c.covers, c.root = fetcher, covers, tmp_path
        yield c


# ---------- search / track ----------

def test_search_by_name_and_artist(client):
    r = client.get("/api/search", params={"q": "ocean drive"}).json()
    assert r[0]["name"] == "Ocean Drive"  # exact match ranks first
    assert all("ocean" in t["name"].lower() and "drive" in t["name"].lower() for t in r)
    # Each word may match the name or the artists; whole-phrase artist matches rank first.
    by_artist = client.get("/api/search", params={"q": "artist 7"}).json()
    phrase = [any(a.lower() == "artist 7" for a in t["artists"]) for t in by_artist]
    assert phrase[0] and phrase == sorted(phrase, reverse=True)
    assert all(any("artist" in a.lower() for a in t["artists"]) and
               ("7" in t["name"] or any("7" in a for a in t["artists"])) for t in by_artist)
    assert len(client.get("/api/search", params={"q": "o"}).json()) == 20  # top 20 only


def test_search_no_results_and_validation(client):
    assert client.get("/api/search", params={"q": "zzzz-nothing"}).json() == []
    assert client.get("/api/search", params={"q": ""}).status_code == 422


def test_featured_keeps_order_and_skips_unknown(engine, settings, tmp_path):
    s = {**settings, "featured": [_id(3), "0" * 22, _id(1)]}
    covers = CoverCache(tmp_path / "covers.sqlite", FakeFetcher())
    app = create_app(engine_factory=lambda _: engine, cover_cache=covers, settings=s,
                     reports_root=tmp_path)
    old = engine.settings
    engine.settings = s
    try:
        with TestClient(app) as c:
            assert [t["id"] for t in c.get("/api/featured").json()] == [_id(3), _id(1)]
    finally:
        engine.settings = old


def test_track_details(client):
    t = client.get(f"/api/track/{_id(3)}").json()
    for k in ["id", "name", "artists", "bpm", "camelot", "camelot_num", "camelot_letter",
              "energy", "valence", "year", "has_style", "tempo_folded"]:
        assert k in t
    assert t["camelot"] == f"{t['camelot_num']}{t['camelot_letter']}"
    assert client.get("/api/track/doesnotexist000000000x").status_code == 404


# ---------- next ----------

def test_next_shape_and_order(client):
    seed = _id(20)
    r = client.post("/api/next", json={"seed_id": seed, "n": 8, "alpha": 0.5}).json()
    items = r["items"]
    assert r["seed"]["id"] == seed and len(items) == 8
    finals = [it["final"] for it in items]
    assert finals == sorted(finals, reverse=True)
    for it in items:
        assert it["track"]["id"] != seed
        assert set(it["transition"]) >= {"score", "harmonic", "tempo", "energy_smoothness",
                                         "timbre", "loudness", "era", "energy_delta"}
        assert it["final"] == pytest.approx(0.5 * it["relevance"]["score"]
                                            + 0.5 * it["transition"]["score"])
    seed_artists = set(r["seed"]["artists"])
    assert not any(seed_artists & set(it["track"]["artists"]) for it in items)
    artists = [a for it in items for a in it["track"]["artists"]]
    assert len(artists) == len(set(artists))  # one track per artist


@pytest.mark.parametrize("alpha, key", [(1.0, "relevance"), (0.0, "transition")])
def test_next_alpha_extremes(client, alpha, key):
    items = client.post("/api/next", json={"seed_id": _id(30), "n": 6, "alpha": alpha}).json()["items"]
    vals = [it[key]["score"] for it in items]
    assert vals == sorted(vals, reverse=True)


def test_next_missing_style_is_null(client):
    items = client.post("/api/next", json={"seed_id": _id(2), "n": 5}).json()["items"]
    assert all(it["relevance"]["style"] is None for it in items)  # seed 2 has no style vector


def test_next_errors(client):
    assert client.post("/api/next", json={"seed_id": "nope"}).status_code == 404
    assert client.post("/api/next", json={"seed_id": _id(1), "alpha": 2}).status_code == 422


# ---------- mix ----------

@pytest.mark.parametrize("generator", [None, "beam_v1", "greedy_v0"])
def test_mix(client, engine, generator):
    arc = [0.3, 0.6, 0.9, 0.5]
    body = {"seed_ids": [_id(40)], "arc": arc, "alpha": 0.4, "length": 10}
    if generator:
        body["generator"] = generator
    r = client.post("/api/mix", json=body).json()
    steps = r["steps"]
    assert r["generator"] == (generator or "beam_v1")  # beam_v1 is the default
    assert r["beam_width"] == (None if generator == "greedy_v0" else engine.gcfg["beam"]["width"])
    assert r["runtime_ms"] > 0 and r["pool_size"] > 10
    assert len(steps) == 10 and len(r["arc"]) == 10
    assert r["arc"][0] == pytest.approx(0.3) and r["arc"][-1] == pytest.approx(0.5)
    assert steps[0]["track"]["id"] == _id(40) and steps[0]["transition"] is None
    ids = [s["track"]["id"] for s in steps]
    assert len(ids) == len(set(ids))
    artists = [a for s in steps for a in s["track"]["artists"]]
    assert len(artists) == len(set(artists))  # max_per_artist = 1
    gamma, lam = r["gamma"], r["weak_link_lambda"]
    tau = engine.gcfg["objective"]["weak_link"]["tau"]
    assert steps[0]["is_seed"] and not any(s["is_seed"] for s in steps[1:])
    for s in steps[1:]:
        T = s["transition"]["score"]
        assert s["arc_fit"] == pytest.approx(1 - abs(s["track"]["energy"] - s["target_energy"]))
        assert s["objective"] == pytest.approx(0.4 * s["relevance"] + 0.6 * T
                                               - lam * max(0, tau - T) + gamma * s["arc_fit"])


def test_mix_best_fit_seed_position(client, engine):
    arc = [0.0, 1.0]  # targets rise linearly, so the seed lands where its energy fits
    r = client.post("/api/mix", json={"seed_ids": [_id(40)], "arc": arc, "length": 10,
                                      "seed_position": "best_fit"}).json()
    steps, k = r["steps"], r["seed_index"]
    assert r["seed_position"] == "best_fit" and len(steps) == 10
    assert steps[k]["track"]["id"] == _id(40) and steps[k]["is_seed"]
    assert k == int(np.argmin([abs(s["target_energy"] - steps[k]["track"]["energy"]) for s in steps]))
    assert steps[0]["transition"] is None and all(s["transition"] for s in steps[1:])
    assert steps[k]["objective"] is None
    gamma, lam = r["gamma"], r["weak_link_lambda"]
    tau = engine.gcfg["objective"]["weak_link"]["tau"]
    for t, s in enumerate(steps):
        if t == k:
            continue
        # tracks before the seed were scored on the transition into their right neighbour
        T = (steps[t + 1] if t < k else s)["transition"]["score"]
        assert s["objective"] == pytest.approx(0.5 * s["relevance"] + 0.5 * T
                                               - lam * max(0, tau - T) + gamma * s["arc_fit"])
    bad = client.post("/api/mix", json={"seed_ids": [_id(40)], "arc": arc, "seed_position": "end"})
    assert bad.status_code == 422


def test_mix_greedy_equals_beam_width_1(engine):
    eng = Engine.__new__(Engine)
    eng.__dict__.update(engine.__dict__)
    eng.gcfg = {**engine.gcfg, "beam": {"width": 1}}
    arc = [0.2, 0.8, 0.4]
    ids = lambda r: [s["track"]["id"] for s in r["steps"]]  # noqa: E731
    assert ids(eng.mix([_id(70)], arc, 0.5, 10, "beam_v1")) == \
        ids(engine.mix([_id(70)], arc, 0.5, 10, "greedy_v0"))


def test_mix_follows_arc_better_with_strong_gamma(engine):
    """With a strong arc term the achieved energy should track a steep target closely."""
    eng = Engine.__new__(Engine)
    eng.__dict__.update(engine.__dict__)
    eng.gcfg = {**engine.gcfg, "objective": {**engine.gcfg["objective"], "gamma": 5.0}}
    arc = [0.1, 0.9, 0.1]
    r = eng.mix([_id(50)], arc, 0.5, 9)
    mae = np.mean([abs(s["track"]["energy"] - s["target_energy"]) for s in r["steps"][1:]])
    assert mae < 0.1


def test_mix_unknown_generator_rejected(client):
    r = client.post("/api/mix", json={"seed_ids": [_id(1)], "arc": [0.5], "generator": "rl"})
    assert r.status_code == 422


def test_mix_validation(client):
    bad_arc = client.post("/api/mix", json={"seed_ids": [_id(1)], "arc": [0.2, 1.4]})
    assert bad_arc.status_code == 422
    assert client.post("/api/mix", json={"seed_ids": ["nope"], "arc": [0.5]}).status_code == 404
    too_long = client.post("/api/mix", json={"seed_ids": [_id(1)], "arc": [0.5], "length": 150})
    assert too_long.status_code == 422


# ---------- covers ----------

def test_cover_cache_hit_and_failures(client):
    ok, missing, flaky = _id(4), _id(5), _id(6)
    first = client.get(f"/api/cover/{ok}").json()
    assert first == {"url": f"https://img.example/{ok}", "cached": False, "status": "ok"}
    assert client.get(f"/api/cover/{ok}").json()["cached"] is True
    assert client.get(f"/api/cover/{missing}").json()["url"] is None
    assert client.get(f"/api/cover/{missing}").json()["cached"] is True  # failure cached
    client.get(f"/api/cover/{flaky}")
    assert client.get(f"/api/cover/{flaky}").json()["cached"] is True  # within TTL
    assert [c[0] for c in client.fetcher.calls] == [ok, missing, flaky]


def test_cover_transient_failure_retried_after_ttl(tmp_path):
    fetcher = FakeFetcher({_id(7): FetchResult(None, "error")})
    cache = CoverCache(tmp_path / "c.sqlite", fetcher, rate_per_s=100, failure_ttl_s=0)

    async def go():
        await cache.get(_id(7))
        await cache.get(_id(7))
    asyncio.run(go())
    assert len(fetcher.calls) == 2


def test_cover_persists_across_instances(tmp_path):
    f1 = FakeFetcher()
    asyncio.run(CoverCache(tmp_path / "c.sqlite", f1, rate_per_s=100).get(_id(8)))
    f2 = FakeFetcher()
    res = asyncio.run(CoverCache(tmp_path / "c.sqlite", f2, rate_per_s=100).get(_id(8)))
    assert res["cached"] is True and f2.calls == []


def test_cover_rate_limited_and_deduplicated(tmp_path):
    fetcher = FakeFetcher()
    cache = CoverCache(tmp_path / "c.sqlite", fetcher, rate_per_s=10)  # >= 0.1 s apart

    async def go():
        await asyncio.gather(*[cache.get(_id(100 + i)) for i in range(4)],
                             cache.get(_id(100)), cache.get(_id(100)))
    asyncio.run(go())
    ids = [c[0] for c in fetcher.calls]
    assert sorted(ids) == sorted({_id(100 + i) for i in range(4)})  # no duplicate fetches
    times = sorted(c[1] for c in fetcher.calls)
    assert all(b - a >= 0.09 for a, b in zip(times, times[1:]))


def test_rate_limiter_spacing():
    async def go():
        lim = RateLimiter(20)
        t = []
        for _ in range(4):
            await lim.wait()
            t.append(time.monotonic())
        return t
    t = asyncio.run(go())
    assert all(b - a >= 0.045 for a, b in zip(t, t[1:]))


def test_cover_invalid_id(client):
    assert client.get("/api/cover/not-a-valid-id").status_code == 400


# ---------- eval ----------

def test_eval_unavailable_then_available(client):
    assert client.get("/api/eval").json()["available"] is False
    assert client.get("/api/figures/pareto_alpha.png").status_code == 404
    rep = client.root / "reports"
    (rep / "figures").mkdir(parents=True)
    pd.DataFrame([{
        "pool": "labeled", "method": m, "alpha": a, "is_d": d, "genre_agree": 0.1,
        "genre_agree_lo": 0.05, "genre_agree_hi": 0.15, "genre_macro": 0.1, "trans": 0.8,
        "trans_lo": 0.79, "trans_hi": 0.81, "dbpm": 2.0, "ild": 0.4, "artist_rep": 0.1,
    } for m, a, d in [("transition", None, False), ("full", 0.5, True), ("full", 0.9, False)]]
    ).to_csv(rep / "eval_v2.csv", index=False)
    (rep / "figures" / "pareto_alpha.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    ev = client.get("/api/eval").json()
    assert ev["available"] is True and len(ev["rows"]) == 2  # sweep rows excluded
    assert ev["figure_url"] == "/api/figures/pareto_alpha.png"
    assert client.get("/api/figures/pareto_alpha.png").status_code == 200
    assert client.get("/api/figures/other.png").status_code == 404
