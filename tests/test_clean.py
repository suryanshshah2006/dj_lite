import numpy as np
import pytest

from djlite.clean import fold_tempo, normalize_name, strip_featuring


# ---------- fold_tempo ----------

@pytest.mark.parametrize("tempo, expected", [
    (80.0, 80.0),
    (120.0, 120.0),
    (159.9, 159.9),
    (160.0, 80.0),
    (175.0, 87.5),
    (200.0, 100.0),
    (320.0, 80.0),
    (60.0, 120.0),
    (40.0, 80.0),
    (30.0, 120.0),
    (79.99, 159.98),
])
def test_fold_tempo_scalar(tempo, expected):
    assert fold_tempo(tempo) == pytest.approx(expected)


def test_fold_tempo_array_in_range():
    t = np.array([31.5, 64.0, 99.0, 140.0, 161.0, 248.9])
    out = fold_tempo(t)
    assert out.shape == t.shape
    assert np.all((out >= 80) & (out < 160))
    # Folding only multiplies by powers of two.
    ratio = np.log2(out / t)
    assert np.allclose(ratio, np.round(ratio))


def test_fold_tempo_custom_range():
    assert fold_tempo(150.0, low=60, high=120) == pytest.approx(75.0)


@pytest.mark.parametrize("bad", [0.0, -120.0, np.nan])
def test_fold_tempo_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        fold_tempo(bad)


def test_fold_tempo_rejects_non_octave_range():
    with pytest.raises(ValueError):
        fold_tempo(120.0, low=80, high=150)


# ---------- normalize_name ----------

@pytest.mark.parametrize("raw, expected", [
    ("Testify", "testify"),
    ("  Hey   Jude  ", "hey jude"),
    ("Hey Jude - Remastered 2015", "hey jude"),
    ("Let It Be (2009 Remaster)", "let it be"),
    ("Underway - 2013 Remaster", "underway"),
    ("Kingston Advice - Remastered", "kingston advice"),
    ("Nocturne 2 (Remastered)", "nocturne 2"),
    ("Song [Remastered]", "song"),
    ("Song - Digital Remaster 2001", "song"),
    ("Song - Remastered 2011 Version", "song"),
    ("Song - 1987 Remastered", "song"),
    ("C'était charmant - Remasterisé en 2016", "c'était charmant"),
    ("Song (Remastered) [feat. X]", "song [feat. x]"),
    ("Romance - Demo Sketch; 2019 Remaster", "romance - demo sketch"),
    ("Song - Live / Remastered", "song - live"),
])
def test_normalize_name_strips_remaster(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize("raw", [
    "Freedom (Live)",
    "Mean To Me - Live",
    "Down - FJAAK Remix",
    "Brittle (Stromkern Remix)",
    "Now or Never - Radio Edit",
    "I Cannot Say (Piano Version)",
    "Song (Acoustic Version)",
    "Song - Acoustic",
    "Oh Sheila (Re-Recorded / Remastered)",
    "Remastered",
    "Remasters of the Universe",
])
def test_normalize_name_keeps_other_tags(raw):
    assert normalize_name(raw) == " ".join(raw.lower().split())


def test_live_and_studio_stay_distinct():
    assert normalize_name("Song - Live") != normalize_name("Song - Remastered")


# ---------- strip_featuring ----------

@pytest.mark.parametrize("raw, expected", [
    ("call me will (feat. nuri olsen)", "call me will"),
    ("song [feat. chris jones]", "song"),
    ("song (ft. x & y)", "song"),
    ("song (featuring kwame toure)", "song"),
    ("song feat. mark da shark", "song"),
    ("say yeah - feat. blaze burna", "say yeah"),
    ("holla like u know me ft. que diesel", "holla like u know me"),
    ("reachin' featuring macklemore and sol", "reachin'"),
    ("song (live) [feat. chris jones]", "song (live)"),
    ("song feat. x - radio edit", "song - radio edit"),
    ("song feat. x (club mix)", "song (club mix)"),
    ("song (feat. x) - remix", "song - remix"),
])
def test_strip_featuring(raw, expected):
    assert strip_featuring(raw) == expected


@pytest.mark.parametrize("raw", [
    "a great feat",
    "feat of strength",
    "left ft right",
    "featuring",
    "song (with teddy wilson)",
    "song - remix",
])
def test_strip_featuring_leaves_other_names(raw):
    assert strip_featuring(raw) == raw
