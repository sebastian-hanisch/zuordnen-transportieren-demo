"""Hausgenerator SplitMix64: bekannte Ausgaben der Referenzimplementierung (Zufalls-Seed 0), Plattformfestigkeit von normal() und expovariate(),
Kopie gleich der Vorlage im Website-Repo (falls vorhanden)."""

import math
from pathlib import Path

import pytest

from ztr_rng import SplitMix64, exact_cos, exact_log

ROOT = Path(__file__).resolve().parent.parent


def test_bekannte_ausgaben_der_referenzimplementierung_fuer_seed_0():
    g = SplitMix64(0)
    assert [g.next_u64() for _ in range(4)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC]


def test_gleicher_seed_gleiche_folge_und_zufall_liegt_im_halboffenen_intervall():
    a, b = SplitMix64(7), SplitMix64(7)
    xs = [a.random() for _ in range(1000)]
    assert xs == [b.random() for _ in range(1000)] and all(0.0 <= x < 1.0 for x in xs)
    assert SplitMix64(7).random() != SplitMix64(8).random()


def test_exponentialzahlen_haben_das_gewuenschte_mittel_und_exakte_hilfsfunktionen_stimmen():
    g = SplitMix64(3)
    xs = [g.expovariate(1.0) for _ in range(4000)]
    assert all(x >= 0 for x in xs) and sum(xs) / len(xs) == pytest.approx(1.0, abs=0.06)       # Mittel 1 mit Standardfehler 1/sqrt(4000) = 0,016
    for x in (0.1, 0.5, 1.0, 2.0, 7.3):
        assert exact_log(x) == pytest.approx(math.log(x), rel=1e-15)
    for x in (0.0, 0.3, 1.0, 3.0, 6.0):
        assert exact_cos(x) == pytest.approx(math.cos(x), abs=1e-15)


def test_kopie_gleich_der_vorlage_im_website_repo_falls_vorhanden():
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / "ztr_rng.py").read_bytes() == vorlage.read_bytes()
