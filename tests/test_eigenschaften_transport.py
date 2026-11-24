"""Eigenschaftstests (metamorph): Aussagen, die für das Transportproblem gelten müssen, ohne dass man das richtige Ergebnis kennt. Je Eigenschaft 20 bis 50 Instanzen
mit festem Zufalls-Seed. Wo der optimale Plan nur bei stetigen Kosten eindeutig ist, laufen die Tests mit „stetig (Zufallskosten)“ oder „Punkte im Einheitsquadrat“;
bei Gleichstand wird nie bitgleich verglichen, sondern geprüft, dass der Plan zur optimalen Fläche gehört (Wert, Randsummen)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ztr_constants as C
import ztr_scenario as SC
import ztr_simplex as S
from ztr_evaluation import eintraege, gespaltene_lager, ist_permutation, mischung
from ztr_rng import SplitMix64

STETIG = ("stetig (Zufallskosten)", "Punkte im Einheitsquadrat")
ROOT = Path(__file__).resolve().parent.parent


def _loese(inst, regel="steilste"):
    return S.loese_transport(inst.a, inst.b, inst.kosten, regel, SplitMix64(1))


@pytest.mark.parametrize("familie", STETIG)
def test_kosten_mit_faktor_multipliziert_oder_um_konstante_verschoben_aendern_den_plan_nicht(familie):
    for k in range(25):
        inst = SC.make_instanz(familie, 5 + k % 6, 5 + k % 6, 1.0, 21, 5, k)
        lo = _loese(inst)
        gross = S.loese_transport(inst.a, inst.b, inst.kosten * 7.5, "steilste")
        assert np.allclose(gross.plan, lo.plan, atol=1e-12) and gross.kosten == pytest.approx(7.5 * lo.kosten, rel=1e-12)
        schief = S.loese_transport(inst.a, inst.b, inst.kosten + 3.0, "steilste")
        assert np.allclose(schief.plan, lo.plan, atol=1e-12)
        assert schief.kosten == pytest.approx(lo.kosten + 3.0 * inst.a.sum(), rel=1e-12)       # die Konstante wirkt bei gleichen Gesamtmassen nur additiv


@pytest.mark.parametrize("familie", STETIG)
def test_lager_und_kunden_umnummerieren_permutiert_den_plan(familie):
    rng = SplitMix64(5)
    for k in range(25):
        inst = SC.make_instanz(familie, 6, 8, 1.0, 22, 5, k)
        zeilen = list(range(6))
        spalten = list(range(8))
        rng.shuffle(zeilen)
        rng.shuffle(spalten)
        lo = _loese(inst)
        neu = S.loese_transport(inst.a[zeilen], inst.b[spalten], inst.kosten[np.ix_(zeilen, spalten)], "steilste")
        assert np.allclose(neu.plan, lo.plan[np.ix_(zeilen, spalten)], atol=1e-12)
        assert neu.kosten == pytest.approx(lo.kosten, rel=1e-12)


@pytest.mark.parametrize("familie", STETIG)
def test_massen_mit_demselben_faktor_multipliziert_skaliert_den_plan(familie):
    for k in range(25):
        inst = SC.make_instanz(familie, 7, 7, 1.0, 23, 5, k)
        lo = _loese(inst)
        gross = S.loese_transport(inst.a * 1000.0, inst.b * 1000.0, inst.kosten, "steilste")
        assert np.allclose(gross.plan, 1000.0 * lo.plan, atol=1e-9) and gross.kosten == pytest.approx(1000.0 * lo.kosten, rel=1e-12)


@pytest.mark.parametrize("familie", C.FAMILIEN)
def test_bei_massen_1_durch_n_ist_jede_ecke_eine_permutation_fuer_jede_regel(familie):
    for k in range(12):
        n = 3 + k % 8
        inst = SC.make_instanz(familie, n, n, 0.0, 24, 5, k)
        for regel in S.ALLE_REGELN:
            lo = S.loese_transport(inst.a, inst.b, inst.kosten, regel, SplitMix64(k))
            assert ist_permutation(lo.plan) and eintraege(lo.plan) == n, (familie, n, regel)


def test_nicht_entartete_ecke_hat_n_plus_m_minus_1_eintraege_und_summe_k_minus_1_gleich_m_minus_1():
    for k in range(50):
        n, m = 12, 7
        inst = SC.make_instanz("Punkte im Einheitsquadrat", n, m, 1.0, 25, 5, k)
        lo = _loese(inst)
        zeilen = (lo.plan > 1e-12).sum(axis=1)
        assert eintraege(lo.plan) == n + m - 1, k
        assert int((zeilen - 1).sum()) == m - 1, k                 # Summe über die Lager der Zahl der Zusatzverbindungen
        assert gespaltene_lager(lo.plan) == int((zeilen > 1).sum())


def test_mischung_zweier_optimaler_plaene_ist_zulaessig_und_gleich_gut():
    gefunden = 0
    for k in range(40):
        inst = SC.make_instanz("Gitter L1", 8, 8, 0.0, 26, 5, k)
        plaene = [S.loese_transport(inst.a, inst.b, inst.kosten, r, SplitMix64(k)).plan for r in S.REGELN]
        wert = (inst.kosten * plaene[0]).sum()
        for p in plaene[1:]:
            if np.abs(p - plaene[0]).max() > 1e-9:
                gefunden += 1
            for t in (0.25, 0.5, 0.75):
                q = mischung(plaene[0], p, t)
                assert (q >= -1e-15).all() and np.allclose(q.sum(axis=1), inst.a) and np.allclose(q.sum(axis=0), inst.b)
                assert (inst.kosten * q).sum() == pytest.approx(wert, rel=1e-12)
    assert gefunden >= 20           # die Eigenschaft wurde an echten Paaren verschiedener optimaler Pläne geprüft


def test_verschiebung_der_preise_laesst_zulaessigkeit_und_dualwert_unveraendert():
    for k in range(25):
        inst = SC.make_instanz("Punkte im Einheitsquadrat", 8, 9, 1.0, 27, 5, k)
        lo = _loese(inst)
        for s in (-2.0, 0.37, 11.0):
            u, v = lo.u + s, lo.v - s
            assert (u[:, None] + v[None, :] <= inst.kosten + 1e-9).all()
            assert inst.a @ u + inst.b @ v == pytest.approx(inst.a @ lo.u + inst.b @ lo.v, rel=1e-12, abs=1e-12)


def test_billigere_zelle_verschlechtert_das_optimum_nie():
    for k in range(40):
        inst = SC.make_instanz("stetig (Zufallskosten)", 6, 6, 1.0, 28, 5, k)
        g = SC.instanz_rng(28, 6, k)
        kosten = inst.kosten.copy()
        kosten[g.below(6), g.below(6)] -= 0.3
        assert S.loese_transport(inst.a, inst.b, kosten, "steilste").kosten <= _loese(inst).kosten + 1e-12


def test_gleicher_zufalls_seed_gleiche_ausgabe_auch_bei_der_zufaelligen_regel():
    inst = SC.make_instanz("Gitter L1", 10, 10, 0.0, 7, 4, 0)
    a = S.loese_transport(inst.a, inst.b, inst.kosten, "zufällige", SplitMix64(3))
    b = S.loese_transport(inst.a, inst.b, inst.kosten, "zufällige", SplitMix64(3))
    assert np.array_equal(a.plan, b.plan) and a.schritte == b.schritte


def test_ergebnis_haengt_nicht_an_der_hash_randomisierung_zweiter_prozess():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import hashlib, ztr_evaluation as E, ztr_constants as C;"
        "a = E.analysiere(E.Settings(quelle=C.QUELLE_ZUFALL, familie='Gitter L1', n=10, seed=7, regel='zufällige'));"
        "g = E.gleichstand_ansicht(E.Settings(quelle=C.QUELLE_ZUFALL, familie='Gitter L1', n=10, seed=7));"
        "print(hashlib.sha256(a.loesung.plan.tobytes() + b''.join(l.plan.tobytes() for l in g.loesungen)).hexdigest())"
    )
    ausgaben = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(ROOT)], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr
        ausgaben.append(r.stdout.strip())
    assert ausgaben[0] == ausgaben[1] and len(ausgaben[0]) == 64
