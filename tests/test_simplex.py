"""Transportsimplex gegen Handrechnung (Handbeispiele 1 bis 4) und gegen die Aufzählung aller Ecken in Brüchen (DEMO-PLAYBOOK §1b und §12).

Die Handwerte sind von Hand nachgerechnet; der Erwartungswert steht als Rechnung im Test, nicht als Abschrift der Ausgabe des Codes."""

import itertools
import random
from fractions import Fraction as F

import numpy as np
import pytest

import ztr_simplex as S
from ecken_aufzaehlung import ecken, optimale_ecken
from ztr_eindeutigkeit import ist_eindeutig
from ztr_evaluation import eintraege, gespaltene_lager, ist_monge, ist_permutation, mischung, nebenmasse
from ztr_rng import SplitMix64


def _hand_plan(zeilen):
    return np.array(zeilen, float)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 1: Zuordnung 3 x 3
# ----------------------------------------------------------------------------------------------------------------------------------

def test_hand1_die_sechs_zuordnungen_kosten_5_6_6_7_9_11(hand1):
    kosten = sorted(sum(hand1.kosten[i][p[i]] for i in range(3)) for p in itertools.permutations(range(3)))
    assert kosten == [5, 6, 6, 7, 9, 11]          # Rechnung: z. B. (1,0,2) kostet 1 + 2 + 2 = 5, (0,1,2) kostet 4 + 0 + 2 = 6


@pytest.mark.parametrize("regel", S.ALLE_REGELN)
def test_hand1_simplex_findet_die_beste_zuordnung(hand1, regel):
    lo = S.loese_transport(hand1.a, hand1.b, hand1.kosten, regel, SplitMix64(1))
    assert lo.kosten == pytest.approx(5.0)
    assert np.array_equal(lo.plan, _hand_plan([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))     # Lager 1 -> Kunde 2, Lager 2 -> Kunde 1, Lager 3 -> Kunde 3
    assert ist_permutation(lo.plan) and eintraege(lo.plan) == 3                         # entartet: n = 3 statt n + m - 1 = 5 Einträge
    assert lo.basis.sum() == 5                                                           # die Basis bleibt trotzdem ein Baum aus 5 Zellen
    assert S.pruefe_loesung(hand1.a, hand1.b, hand1.kosten, lo) == []


def test_hand1_dualwert_ist_der_primalwert(hand1):
    lo = S.loese_transport(hand1.a, hand1.b, hand1.kosten, "erste")
    assert hand1.a @ lo.u + hand1.b @ lo.v == pytest.approx(5.0)


def test_hand1_ecken_sind_genau_die_sechs_permutationsmatrizen(hand1):
    alle = ecken([1, 1, 1], [1, 1, 1])
    assert len(alle) == 6 and all(sorted(sum(r) for r in p) == [1, 1, 1] and all(x in (0, 1) for r in p for x in r) for p in alle)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 2: Transport 3 x 3 mit ungleichen Mengen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_hand2_kosten_sind_quadrierte_abstaende(hand2):
    assert hand2.kosten.tolist() == [[4, 20, 29], [20, 4, 29], [5, 5, 4]]      # (0,0)-(0,2): 0 + 4; (0,0)-(4,2): 16 + 4; (0,0)-(2,5): 4 + 25; ...


@pytest.mark.parametrize("regel", S.ALLE_REGELN)
def test_hand2_optimum_97_mit_zwei_gespaltenen_lagern(hand2, regel):
    lo = S.loese_transport(hand2.a, hand2.b, hand2.kosten, regel, SplitMix64(1))
    assert lo.kosten == pytest.approx(97.0)                                      # 8·4 + 1·20 + 6·4 + 1·5 + 4·4 = 32 + 20 + 24 + 5 + 16
    assert np.array_equal(lo.plan, _hand_plan([[8, 1, 0], [0, 6, 0], [0, 1, 4]]))
    assert eintraege(lo.plan) == 5 == 3 + 3 - 1                                  # nicht entartet: n + m - 1 Einträge
    assert gespaltene_lager(lo.plan) == 2                                        # Lager 1 und Lager 3
    assert nebenmasse(lo.plan) == pytest.approx(2 / 20)                          # 1 + 1 von 20 Einheiten liegen außerhalb des Hauptabnehmers
    assert not ist_monge(lo.plan)
    zeilenzahl = [int((r > 0).sum()) for r in lo.plan]
    assert sum(k - 1 for k in zeilenzahl) == 3 - 1                               # Summe (k_i - 1) = m - 1 bei einer nicht entarteten Ecke
    assert S.pruefe_loesung(hand2.a, hand2.b, hand2.kosten, lo) == []


def test_hand2_keine_der_27_abbildungen_hat_das_bildmass_der_nachfragen(hand2):
    treffer = 0
    for T in itertools.product(range(3), repeat=3):
        bild = [0, 0, 0]
        for i, j in enumerate(T):
            bild[j] += int(hand2.a[i])
        treffer += bild == [8, 8, 4]
    assert treffer == 0


def test_hand2_die_18_ecken_haben_alle_fuenf_eintraege_und_das_optimum_ist_eindeutig(hand2):
    alle = ecken([9, 6, 5], [8, 8, 4])
    assert len(alle) == 18 and all(sum(1 for r in p for x in r if x != 0) == 5 for p in alle)
    opt, beste = optimale_ecken([9, 6, 5], [8, 8, 4], hand2.kosten.astype(int).tolist())
    assert opt == 97 and len(beste) == 1


def test_hand2_preise_schlupf_und_dualwert(hand2):
    lo = S.loese_transport(hand2.a, hand2.b, hand2.kosten, "erste")
    assert lo.u.tolist() == [0, -16, -15] and lo.v.tolist() == [4, 20, 19]       # von Hand: u1 = 0, v1 = 4, v2 = 20, u2 = 4 - 20 = -16, u3 = 5 - 20 = -15, v3 = 4 + 15 = 19
    schlupf = S.reduzierte_kosten(hand2.kosten, lo.u, lo.v)
    assert schlupf.tolist() == [[0, 0, 10], [32, 0, 26], [16, 0, 0]]
    assert ((schlupf == 0) == (lo.plan > 0)).all()                               # komplementärer Schlupf: null genau auf dem Träger
    assert hand2.a @ lo.u + hand2.b @ lo.v == 97                                 # 9·0 + 6·(-16) + 5·(-15) + 8·4 + 8·20 + 4·19 = -171 + 268


def test_hand2_preise_gelten_nur_bis_auf_eine_konstante(hand2):
    lo = S.loese_transport(hand2.a, hand2.b, hand2.kosten, "erste")
    for s in (-3.0, 0.5, 7.0):
        u, v = lo.u + s, lo.v - s
        assert (hand2.kosten - u[:, None] - v[None, :] >= -1e-12).all()
        assert hand2.a @ u + hand2.b @ v == pytest.approx(97.0)


def test_hand2_rang_der_tragergleichungen_ist_5_bei_6_unbekannten(hand2):
    lo = S.loese_transport(hand2.a, hand2.b, hand2.kosten, "erste")
    zellen = np.argwhere(lo.plan > 0)
    M = np.zeros((len(zellen), 6))
    for r, (i, j) in enumerate(zellen):
        M[r, i] = 1.0
        M[r, 3 + j] = 1.0
    assert np.linalg.matrix_rank(M) == 5


def test_hand2_preisprobe_eine_einheit_mehr_bei_lager_3_und_kunde_1(hand2):
    lo = S.loese_transport(hand2.a, hand2.b, hand2.kosten, "erste")
    neu = S.loese_transport([9, 6, 6], [9, 8, 4], hand2.kosten, "erste")
    assert neu.kosten == pytest.approx(86.0)                                      # 97 + u3 + v1 = 97 - 15 + 4
    assert neu.kosten - lo.kosten == pytest.approx(lo.u[2] + lo.v[0]) == pytest.approx(-11.0)
    opt, _ = optimale_ecken([9, 6, 6], [9, 8, 4], hand2.kosten.astype(int).tolist())
    assert opt == 86                                                              # unabhängig durch erneutes Aufzählen aller Ecken
    assert len(ecken([9, 6, 6], [9, 8, 4])) == 16                                 # das Polytop der geänderten Aufgabe hat 16 Ecken (die 18 gehören zur ursprünglichen Aufgabe)


def test_hand2_gegenstueck_gleiche_massen_8_8_4_geben_den_diagonalplan_ohne_spaltung(hand2):
    lo = S.loese_transport([8, 8, 4], [8, 8, 4], hand2.kosten, "erste")
    assert lo.kosten == pytest.approx(80.0)                                       # 8·4 + 8·4 + 4·4
    assert np.array_equal(lo.plan, np.diag([8.0, 8.0, 4.0]))
    assert ist_monge(lo.plan) and eintraege(lo.plan) == 3 < 5                     # ungleiche Massen (nicht 1/n) und trotzdem keine Spaltung; entartet
    assert ist_eindeutig(hand2.kosten, lo.plan, lo.u, lo.v)


def test_hand2_ergaenzung_massen_1_2_gegen_2_1_haben_eine_monge_abbildung():
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    lo = S.loese_transport([1, 2], [2, 1], C, "erste")
    assert lo.kosten == pytest.approx(0.0)                                         # Lager 1 -> Kunde 2 (1 Einheit), Lager 2 -> Kunde 1 (2 Einheiten)
    assert np.array_equal(lo.plan, _hand_plan([[0, 1], [2, 0]]))
    assert ist_monge(lo.plan) and eintraege(lo.plan) == 2 < 3
    assert len(ecken([1, 2], [2, 1])) == 2                                         # das Polytop hat genau zwei Ecken (Kosten 0 und 2)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 3: Gleichstand, der optimale Plan ist eine Fläche
# ----------------------------------------------------------------------------------------------------------------------------------

LAGER3 = [(0, 0), (1, 1), (5, 5)]
KUNDEN3 = [(0, 1), (1, 0), (5, 6)]
C3 = np.array([[abs(l[0] - k[0]) + abs(l[1] - k[1]) for k in KUNDEN3] for l in LAGER3], float)
D3 = np.array([[(l[0] - k[0]) ** 2 for k in KUNDEN3] for l in LAGER3], float)      # zweite Größe: quadrierte waagerechte Verschiebung


def test_hand3_zwei_beste_zuordnungen_mit_kosten_3():
    assert C3.tolist() == [[1, 1, 11], [1, 1, 9], [9, 9, 1]]
    kosten = {p: sum(C3[i][p[i]] for i in range(3)) for p in itertools.permutations(range(3))}
    assert sorted(kosten.values()) == [3, 3, 19, 19, 21, 21]
    assert sorted(p for p, c in kosten.items() if c == 3) == [(0, 1, 2), (1, 0, 2)]


@pytest.mark.parametrize("regel", S.REGELN)
def test_hand3_jede_regel_liefert_einen_der_beiden_besten_plaene(regel):
    lo = S.loese_transport(np.ones(3), np.ones(3), C3, regel, SplitMix64(4))
    assert lo.kosten == pytest.approx(3.0)
    p1 = np.eye(3)
    p2 = np.eye(3)[[1, 0, 2]]
    assert np.array_equal(lo.plan, p1) or np.array_equal(lo.plan, p2)


def test_hand3_nicht_eindeutig_und_jede_mischung_ist_optimal_und_zulaessig():
    lo = S.loese_transport(np.ones(3), np.ones(3), C3, "erste")
    assert not ist_eindeutig(C3, lo.plan, lo.u, lo.v)
    p1, p2 = np.eye(3), np.eye(3)[[1, 0, 2]]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = mischung(p1, p2, t)
        assert (C3 * q).sum() == pytest.approx(3.0)
        assert q.sum(axis=0).tolist() == [1, 1, 1] and q.sum(axis=1).tolist() == [1, 1, 1]
    mitte = mischung(p1, p2, 0.5)
    assert not ist_permutation(mitte) and 0.5 in mitte


def test_hand3_zweite_kennzahl_streut_bei_gleichen_kosten():
    p1, p2 = np.eye(3), np.eye(3)[[1, 0, 2]]
    zweit = [(D3 * p).sum() for p in (p1, p2, mischung(p1, p2, 0.5))]
    assert zweit == [0.0, 2.0, 1.0]                                                # 0 + 0 + 0; 1 + 1 + 0; Mittel 1


def test_hand3_aufzaehlung_zeigt_genau_zwei_optimale_ecken():
    opt, beste = optimale_ecken([1, 1, 1], [1, 1, 1], C3.astype(int).tolist())
    assert opt == 3 and len(beste) == 2


# ----------------------------------------------------------------------------------------------------------------------------------
# Gegenprobe über viele kleine ganzzahlige Instanzen: Optimalwert und Eindeutigkeit gegen die Aufzählung aller Ecken in Brüchen
# ----------------------------------------------------------------------------------------------------------------------------------

def _kleine_instanz(seed):
    rnd = random.Random(seed)
    n, m = rnd.randint(2, 3), rnd.randint(2, 3)
    a = [rnd.randint(0, 4) for _ in range(n)]
    gesamt = sum(a) or 1
    if sum(a) == 0:
        a[0] = 1
        gesamt = 1
    schnitte = sorted(rnd.randint(0, gesamt) for _ in range(m - 1))
    b = [y - x for x, y in zip([0] + schnitte, schnitte + [gesamt])]
    C = [[rnd.randint(0, 2) for _ in range(m)] for _ in range(n)]
    return a, b, C


@pytest.mark.parametrize("regel", S.ALLE_REGELN)
def test_optimalwert_und_eindeutigkeit_stimmen_mit_der_aufzaehlung_aller_ecken(regel):
    rng = SplitMix64(9)
    mehrere = 0
    for seed in range(300):
        a, b, C = _kleine_instanz(seed)
        opt, beste = optimale_ecken(a, b, C)
        lo = S.loese_transport(a, b, C, regel, rng)
        assert lo.kosten == pytest.approx(float(opt), abs=1e-9), (seed, a, b, C)
        plan = tuple(tuple(F(int(round(x))) for x in z) for z in lo.plan)
        assert plan in beste, (seed, a, b, C)                                      # die Ecke gehört zu den optimalen Ecken (Plan ganzzahlig, da Mengen ganzzahlig)
        assert ist_eindeutig(np.array(C, float), lo.plan, lo.u, lo.v) == (len(beste) == 1), (seed, a, b, C)
        mehrere += len(beste) > 1
    assert mehrere >= 25          # die Stichprobe enthält genug Gleichstände, damit der Vergleich der Eindeutigkeit etwas prüft


# ----------------------------------------------------------------------------------------------------------------------------------
# Randfälle
# ----------------------------------------------------------------------------------------------------------------------------------

def test_ein_lager_oder_ein_kunde_der_plan_ist_die_vorgabe():
    lo = S.loese_transport([5.0], [2.0, 3.0], [[1.0, 4.0]], "erste")
    assert lo.plan.tolist() == [[2.0, 3.0]] and lo.kosten == pytest.approx(14.0)
    lo = S.loese_transport([2.0, 3.0], [5.0], [[1.0], [4.0]], "letzte")
    assert lo.plan.tolist() == [[2.0], [3.0]] and lo.kosten == pytest.approx(14.0)
    lo = S.loese_transport([7.0], [7.0], [[3.0]], "zufällige", SplitMix64(1))
    assert lo.plan.tolist() == [[7.0]] and lo.schritte == 0


def test_lager_mit_menge_0_wird_nicht_beliefert_und_die_basis_bleibt_ein_baum():
    C = np.array([[1.0, 2.0], [0.0, 0.0], [2.0, 1.0]])
    lo = S.loese_transport([1.0, 0.0, 1.0], [1.0, 1.0], C, "erste")
    assert lo.plan.sum(axis=1).tolist() == [1.0, 0.0, 1.0] and lo.kosten == pytest.approx(2.0)
    assert lo.basis.sum() == 3 + 2 - 1
    assert S.pruefe_loesung([1.0, 0.0, 1.0], [1.0, 1.0], C, lo) == []


def test_alle_kosten_gleich_die_ganze_flaeche_ist_optimal():
    C = np.full((3, 3), 2.0)
    lo = S.loese_transport(np.ones(3), np.ones(3), C, "erste")
    assert lo.kosten == pytest.approx(6.0) and lo.schritte == 0
    assert not ist_eindeutig(C, lo.plan, lo.u, lo.v)
    lo1 = S.loese_transport([4.0], [4.0], C[:1, :1], "erste")
    assert ist_eindeutig(C[:1, :1], lo1.plan, lo1.u, lo1.v)


def test_sehr_kleine_massen_sind_kein_problem_fuer_den_eigenen_simplex():
    a = np.array([1e-9, 3e-9, 2e-9])
    b = np.array([2e-9, 2e-9, 2e-9])
    C = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    lo = S.loese_transport(a, b, C, "erste")
    assert lo.plan.sum() == pytest.approx(6e-9) and lo.kosten > 0                  # ein LP-Löser mit absoluter Toleranz 1e-7 meldet hier „infeasible“


def test_ungleiche_gesamtmengen_geben_einen_klaren_fehler():
    with pytest.raises(ValueError, match="verschieden"):
        S.loese_transport([1.0, 2.0], [1.0, 1.0], np.ones((2, 2)), "erste")


@pytest.mark.parametrize("a,b,C", [([-1.0, 2.0], [0.5, 0.5], np.ones((2, 2))), ([1.0], [1.0], np.ones((2, 2))), ([], [], np.ones((0, 0)))])
def test_ungueltige_eingaben_werden_abgelehnt(a, b, C):
    with pytest.raises(ValueError):
        S.loese_transport(a, b, C, "erste")


def test_unbekannte_regel_und_fehlender_generator():
    with pytest.raises(ValueError, match="unbekannte"):
        S.loese_transport([1.0], [1.0], [[1.0]], "beste")
    with pytest.raises(ValueError, match="Generator"):
        S.loese_transport(np.ones(2), np.ones(2), [[4.0, 1.0], [1.0, 4.0]], "zufällige")


def test_entartete_zuordnungen_kreisen_nicht_und_der_rueckfall_greift_nur_bei_zufaellig_und_steilste():
    rng = SplitMix64(3)
    for seed in range(15):
        g = SplitMix64(seed)
        C = np.array([[float(g.below(3)) for _ in range(12)] for _ in range(12)])            # viele Gleichstände, stark entartet
        for regel in S.ALLE_REGELN:
            lo = S.loese_transport(np.ones(12), np.ones(12), C, regel, rng)
            assert S.pruefe_loesung(np.ones(12), np.ones(12), C, lo) == []
            assert lo.entartete_schritte <= lo.schritte
            if regel in ("erste", "letzte"):
                assert lo.bland_schritte == 0


def test_pruefer_findet_verletzungen():
    C = np.array([[1.0, 2.0], [2.0, 1.0]])
    lo = S.loese_transport([1.0, 1.0], [1.0, 1.0], C, "erste")
    falsch = S.Loesung(np.array([[0.0, 1.0], [1.0, 0.0]]), 4.0, lo.u, lo.v, lo.basis, 0, 0, 0)
    meldungen = " | ".join(S.pruefe_loesung([1.0, 1.0], [1.0, 1.0], C, falsch))
    assert "positiven reduzierten Kosten" in meldungen and "Primalwert und Dualwert" in meldungen
    schief = S.Loesung(np.array([[2.0, 0.0], [0.0, 1.0]]), 3.0, lo.u, lo.v, lo.basis, 0, 0, 0)
    assert "Zeilensummen" in " | ".join(S.pruefe_loesung([1.0, 1.0], [1.0, 1.0], C, schief))
    minus = S.Loesung(np.array([[2.0, -1.0], [-1.0, 2.0]]), 0.0, lo.u, lo.v, lo.basis, 0, 0, 0)
    assert "negative Menge" in " | ".join(S.pruefe_loesung([1.0, 1.0], [1.0, 1.0], C, minus))


# ----------------------------------------------------------------------------------------------------------------------------------
# Einheiten des Simplex an der Mini-Instanz
# ----------------------------------------------------------------------------------------------------------------------------------

def test_nordwest_ecke_fuellt_von_links_oben_und_bleibt_ein_baum_auch_bei_gleichzeitig_leeren_zeile_und_spalte():
    x, basis = S.nordwest_ecke([9, 6, 5], [8, 8, 4])
    assert x.tolist() == [[8, 1, 0], [0, 6, 0], [0, 1, 4]] and basis.sum() == 5          # 8 | 1 | 6 | 1 | 4: die Plan-Ecke von Hand ist hier schon die Startecke
    x, basis = S.nordwest_ecke([1, 1, 1], [1, 1, 1])                                    # Zeile und Spalte werden zugleich leer: die Diagonale plus Zellen mit Menge 0
    assert x.tolist() == np.eye(3).tolist() and basis.sum() == 5


def test_potentiale_loesen_u_plus_v_gleich_c_auf_der_basis():
    x, basis = S.nordwest_ecke([9, 6, 5], [8, 8, 4])
    C = np.array([[4.0, 20.0, 29.0], [20.0, 4.0, 29.0], [5.0, 5.0, 4.0]])
    u, v, eltern, tiefe = S.potentiale(basis, C)
    assert u[0] == 0 and (u[:, None] + v[None, :])[basis].tolist() == C[basis].tolist()


def test_kreis_der_eintrittszelle_hat_abwechselnde_vorzeichen_und_schliesst_sich():
    x, basis = S.nordwest_ecke([9, 6, 5], [8, 8, 4])
    C = np.array([[4.0, 20.0, 29.0], [20.0, 4.0, 29.0], [5.0, 5.0, 4.0]])
    u, v, eltern, tiefe = S.potentiale(basis, C)
    kreis = S.kreis_zellen(3, 0 * 3 + 2, eltern, tiefe, 3)          # Eintrittszelle (Lager 1, Kunde 3) lohnt sich nicht, aber der Kreis ist trotzdem zu bilden
    zellen = [z for z, _ in kreis]
    vorzeichen = [s for _, s in kreis]
    assert zellen[0] == (0, 2) and vorzeichen == [(+1 if k % 2 == 0 else -1) for k in range(len(kreis))]
    zeilen = sorted(z[0] for z in zellen)
    spalten = sorted(z[1] for z in zellen)
    assert all(zeilen.count(i) == 2 for i in set(zeilen)) and all(spalten.count(j) == 2 for j in set(spalten))      # jede Zeile und Spalte des Kreises genau zweimal


def test_ein_pivot_aendert_die_kosten_um_schrittweite_mal_reduzierte_kosten():
    C = np.array([[4.0, 1.0], [1.0, 4.0]])      # Start: Diagonale (Kosten 8); Eintritt (0, 1) mit d = -6; Schrittweite 1; danach Kosten 2
    x, basis = S.nordwest_ecke([1.0, 1.0], [1.0, 1.0])
    u, v, eltern, tiefe = S.potentiale(basis, C)
    d = S.reduzierte_kosten(C, u, v)
    eintritt = S.waehle_eintritt(d, basis, "erste", S.toleranz(C))
    kreis = S.kreis_zellen(2, eintritt, eltern, tiefe, 2)
    vorher = float((C * x).sum())
    theta, _ = S.pivotiere(x, basis, kreis, "erste", 2)
    assert eintritt == 1 and d.ravel()[eintritt] == pytest.approx(-6.0)                 # 1 - 0 - 7 = -6
    assert theta == pytest.approx(1.0) and vorher == pytest.approx(8.0)
    assert float((C * x).sum()) == pytest.approx(vorher + theta * d.ravel()[eintritt]) == pytest.approx(2.0)


def test_eintrittsregeln_waehlen_die_erste_letzte_und_eine_zufaellige_der_lohnenden_zellen():
    d = np.array([[0.0, -1.0, 0.5], [-2.0, 0.0, -3.0]])
    basis = np.zeros((2, 3), bool)
    assert S.waehle_eintritt(d, basis, "erste", 1e-9) == 1
    assert S.waehle_eintritt(d, basis, "letzte", 1e-9) == 5
    assert S.waehle_eintritt(d, basis, "steilste", 1e-9) == 5
    gewaehlt = {S.waehle_eintritt(d, basis, "zufällige", 1e-9, SplitMix64(s)) for s in range(40)}
    assert gewaehlt <= {1, 3, 5} and len(gewaehlt) >= 2
    basis[1, 2] = True                                                   # Basiszellen treten nie ein
    assert S.waehle_eintritt(d, basis, "letzte", 1e-9) == 3
    assert S.waehle_eintritt(np.zeros((2, 2)), np.zeros((2, 2), bool), "erste", 1e-9) is None


def test_regeln_unterscheiden_sich_auf_mindestens_einer_instanz():
    unterschiedlich = set()
    for seed in range(30):
        g = SplitMix64(seed)
        C = np.array([[float(g.below(3)) for _ in range(6)] for _ in range(6)])
        plaene = {r: S.loese_transport(np.ones(6), np.ones(6), C, r, SplitMix64(seed)).plan.tobytes() for r in S.REGELN}
        if len(set(plaene.values())) > 1:
            unterschiedlich.add(seed)
    assert len(unterschiedlich) >= 5               # Zweig-Test: eine Regel, die nie anders wählt, wäre ein toter Zweig
