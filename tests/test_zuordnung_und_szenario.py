"""Ungarische Methode gegen Handrechnung und Aufzählung; Zufallsinstanzen: Bestimmtheit, Massen, Gitter und Ziehung (DEMO-PLAYBOOK §1b)."""

import itertools
import math

import numpy as np
import pytest

import ztr_constants as C
import ztr_scenario as SC
from ztr_zuordnung import ungarisch


# ----------------------------------------------------------------------------------------------------------------------------------
# Ungarische Methode
# ----------------------------------------------------------------------------------------------------------------------------------

def test_ungarisch_handbeispiel_kosten_5(hand1):
    zuordnung, kosten = ungarisch(hand1.kosten)
    assert zuordnung.tolist() == [1, 0, 2] and kosten == 5.0          # Lager 1 -> Kunde 2 (1), Lager 2 -> Kunde 1 (2), Lager 3 -> Kunde 3 (2)


def test_ungarisch_gleich_aufzaehlung_aller_permutationen_auf_60_instanzen():
    for k in range(60):
        n = 1 + k % 6
        inst = SC.make_instanz(C.FAMILIEN[k % 7], n, n, 0.0, 5, 11, k)
        brute = min(sum(inst.kosten[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        zuordnung, kosten = ungarisch(inst.kosten)
        assert kosten == pytest.approx(brute, abs=1e-12), (k, n)
        assert sorted(zuordnung.tolist()) == list(range(n))            # eine Permutation


def test_ungarisch_randfaelle():
    assert ungarisch([[3.0]])[0].tolist() == [0]
    zuordnung, kosten = ungarisch(np.full((4, 4), 2.0))                # alle Kosten gleich: irgendeine Permutation, Kosten 8
    assert kosten == 8.0 and sorted(zuordnung.tolist()) == [0, 1, 2, 3]
    assert ungarisch([[-1.0, 2.0], [3.0, -5.0]])[1] == -6.0            # negative Kosten sind erlaubt
    for schlecht in (np.ones((2, 3)), np.ones((0, 0)), np.ones(3)):
        with pytest.raises(ValueError):
            ungarisch(schlecht)


# ----------------------------------------------------------------------------------------------------------------------------------
# Szenario
# ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("familie", C.FAMILIEN)
def test_gleiche_eingabe_gleiche_instanz_und_anderer_index_andere_instanz(familie):
    a = SC.make_instanz(familie, 6, 6, 1.0, 5, 1, 0)
    b = SC.make_instanz(familie, 6, 6, 1.0, 5, 1, 0)
    c = SC.make_instanz(familie, 6, 6, 1.0, 5, 1, 1)
    d = SC.make_instanz(familie, 6, 6, 1.0, 6, 1, 0)
    assert np.array_equal(a.kosten, b.kosten) and np.array_equal(a.a, b.a) and np.array_equal(a.zweit, b.zweit)
    assert not np.array_equal(a.kosten, c.kosten) and not np.array_equal(a.kosten, d.kosten)


def test_massen_haben_summe_1_und_s_0_gibt_gleiche_massen():
    for s in C.S_STUFEN:
        inst = SC.make_instanz("Punkte im Einheitsquadrat", 9, 7, s, 2, 1, 3)
        assert inst.a.sum() == pytest.approx(1.0, abs=1e-12) and inst.b.sum() == pytest.approx(1.0, abs=1e-12)
        assert (inst.a > 0).all() and (inst.b > 0).all()
    gleich = SC.make_instanz("stetig (Zufallskosten)", 8, 8, 0.0, 2, 1, 0)
    assert np.array_equal(gleich.a, np.full(8, 0.125)) and np.array_equal(gleich.b, np.full(8, 0.125))


def test_massen_mit_s_0_001_sind_fast_gleich_gross_und_mit_s_1_deutlich_verschieden():
    klein = SC.make_instanz("stetig (Zufallskosten)", 20, 20, 0.001, 4, 1, 0)
    gross = SC.make_instanz("stetig (Zufallskosten)", 20, 20, 1.0, 4, 1, 0)
    assert np.abs(klein.a - 0.05).max() <= 0.001 * 1.0               # a = (1 - s)/n + s · Dirichlet: Abstand zu 1/n höchstens s
    assert gross.a.max() / gross.a.min() > 3


def test_punkte_im_einheitsquadrat_haben_groesstes_kostenelement_1_und_orte():
    inst = SC.make_instanz("Punkte im Einheitsquadrat", 7, 5, 1.0, 1, 1, 0)
    assert inst.kosten.max() == pytest.approx(1.0) and inst.kosten.shape == (7, 5)
    assert inst.lager.shape == (7, 2) and inst.kunden.shape == (5, 2) and (0 <= inst.lager).all() and (inst.lager <= 1).all()
    assert inst.zweit.max() == pytest.approx(1.0) and not np.allclose(inst.kosten, inst.zweit)


def test_ganzzahlige_familien_haben_ihren_bereich_und_beide_matrizen_sind_verschieden():
    for fam, hoechst in (("ganzzahlig 0 bis 100", 100), ("ganzzahlig 0 bis 10", 10), ("ganzzahlig 0 bis 2", 2)):
        inst = SC.make_instanz(fam, 15, 15, 0.0, 3, 1, 0)
        assert inst.kosten.min() >= 0 and inst.kosten.max() <= hoechst and (inst.kosten == np.round(inst.kosten)).all()
        assert set(np.unique(inst.kosten)) == set(range(hoechst + 1)) or hoechst == 100          # bei 0 bis 10 und 0 bis 2 kommen bei 225 Zellen alle Werte vor
        assert not np.array_equal(inst.kosten, inst.zweit)


def test_gitterseite_standard_ist_wurzel_aus_n_plus_m_aufgerundet_plus_1():
    assert SC.gitter_seite(10, 10, C.GITTER_STANDARD, C.ZIEHUNGEN[0]) == 6      # ceil(sqrt(20)) + 1 = 5 + 1
    assert SC.gitter_seite(30, 30, C.GITTER_STANDARD, C.ZIEHUNGEN[0]) == 9      # ceil(sqrt(60)) + 1 = 8 + 1
    assert SC.gitter_seite(10, 10, 20, C.ZIEHUNGEN[1]) == 20                    # gewünschte Seite, mit Zurücklegen unverändert
    assert SC.gitter_seite(40, 40, 5, C.ZIEHUNGEN[0]) == math.ceil(math.sqrt(80))      # ohne Zurücklegen mindestens so groß, dass n + m verschiedene Punkte Platz haben
    assert SC.gitter_seite(40, 40, 5, C.ZIEHUNGEN[1]) == 5


@pytest.mark.parametrize("familie", C.GITTER_FAMILIEN)
def test_gitter_ohne_zuruecklegen_alle_orte_verschieden_mit_zuruecklegen_kommen_doppelte_vor(familie):
    for k in range(30):
        inst = SC.make_instanz(familie, 10, 10, 0.0, 8, 1, k)
        punkte = {tuple(p) for p in np.vstack([inst.lager, inst.kunden])}
        assert len(punkte) == 20 and inst.gitter_seite == 6
        assert (inst.lager >= 0).all() and (inst.lager < 6).all() and (inst.kosten == np.round(inst.kosten)).all()
    doppelt = 0
    for k in range(30):
        inst = SC.make_instanz(familie, 10, 10, 0.0, 8, 1, k, gitter=5, ziehung=C.ZIEHUNGEN[1])
        doppelt += len({tuple(p) for p in np.vstack([inst.lager, inst.kunden])}) < 20
    assert doppelt >= 25                                                          # 20 Punkte auf 25 Feldern: fast immer doppelte Orte


def test_gitterkosten_stimmen_mit_den_orten_ueberein():
    l1 = SC.make_instanz("Gitter L1", 5, 5, 0.0, 1, 1, 0)
    q = SC.make_instanz("Gitter L2²", 5, 5, 0.0, 1, 1, 0)
    for inst, norm in ((l1, lambda d: np.abs(d).sum()), (q, lambda d: (d ** 2).sum())):
        for i in range(5):
            for j in range(5):
                assert inst.kosten[i, j] == norm(inst.lager[i] - inst.kunden[j])
    for i in range(5):                                                            # zweite Kennzahl: das jeweils andere Abstandsmaß
        for j in range(5):
            assert l1.zweit[i, j] == ((l1.lager[i] - l1.kunden[j]) ** 2).sum()
            assert q.zweit[i, j] == np.abs(q.lager[i] - q.kunden[j]).sum()


def test_handbeispiele_haben_die_werte_der_handrechnung(hand1, hand2):
    assert hand1.kosten.tolist() == [[4, 1, 3], [2, 0, 5], [3, 2, 2]] and hand1.a.tolist() == [1, 1, 1]
    assert hand2.a.tolist() == [9, 6, 5] and hand2.b.tolist() == [8, 8, 4] and hand2.lager is not None


def test_ungueltige_eingaben():
    with pytest.raises(ValueError):
        SC.make_instanz("unbekannt", 3, 3, 0.0, 1, 1, 0)
    with pytest.raises(ValueError):
        SC.make_instanz("Gitter L1", 0, 3, 0.0, 1, 1, 0)
