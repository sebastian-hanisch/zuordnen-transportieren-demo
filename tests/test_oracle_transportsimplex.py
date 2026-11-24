"""Orakel für den Transportsimplex, die Ungarische Methode, die Preise und den Eindeutigkeitstest: jeweils ein ANDERER Rechenweg als der der Demo.

- Optimalwert: `scipy.optimize.linprog` (HiGHS) und `networkx.network_simplex` und der Mindestkostenfluss von OR-Tools (ganzzahlige Instanzen: Abweichung 0;
  stetige: 1e-9), bei gleichen Massen zusätzlich `scipy.optimize.linear_sum_assignment`.
- Dualität: das Dual-LP unabhängig mit `linprog` gelöst (gleicher Wert, die Preise der Demo sind dafür zulässig) und die c-Transformierte v_j = min_i (c_ij - u_i).
- Eindeutigkeit: gegen die Aufzählung aller Permutationen (n = 5, Kosten 0 bis 2) und gegen den eigenen Test über ein zweites Linearprogramm (Richtung auf der optimalen Fläche).
- Welche Ecke ein Löser bei Gleichstand wählt, wird nie bitgleich verglichen: geprüft wird, dass der Plan zur optimalen Fläche gehört (Wert, Randsummen, Preise zulässig).
Fehlt eine Bibliothek im Test-venv, überspringt sich der jeweilige Test (`pytest.importorskip`); alle Läufe zusammen unter 10 Sekunden."""

import itertools

import numpy as np
import pytest

import ztr_constants as C
import ztr_scenario as SC
import ztr_simplex as S
from ztr_eindeutigkeit import ist_eindeutig
from ztr_zuordnung import ungarisch

scipy_opt = pytest.importorskip("scipy.optimize")


def _lp(a, b, Cm, methode="highs-ds"):
    n, m = Cm.shape
    A = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    skala = 1e6 if np.sum(a) <= 1.5 else 1.0       # HiGHS meldet bei absoluten Massen unter etwa 1e-7 „infeasible“: Massen skalieren
    r = scipy_opt.linprog(Cm.ravel(), A_eq=A, b_eq=np.concatenate([a, b]) * skala, bounds=(0, None), method=methode)
    assert r.status == 0, r.message
    return r.fun / skala


def _instanzen(anzahl, n_bis=12, start=0):
    for k in range(anzahl):
        fam = C.FAMILIEN[k % len(C.FAMILIEN)]
        n = 3 + (k * 5 + start) % (n_bis - 2)
        s = (0.0, 0.1, 1.0)[k % 3]
        yield SC.make_instanz(fam, n, n + (k % 3), s, 100 + start, 77, k)


def test_optimalwert_gleich_linprog_highs_auf_60_instanzen_aller_familien():
    for inst in _instanzen(60):
        for regel in S.ALLE_REGELN:
            lo = S.loese_transport(inst.a, inst.b, inst.kosten, regel, SC.instanz_rng(1, 2, 3))
            assert lo.kosten == pytest.approx(_lp(inst.a, inst.b, inst.kosten), rel=1e-9, abs=1e-12), (inst.familie, inst.n, inst.m, regel)


def test_optimalwert_gleich_innere_punkte_von_highs():
    for inst in list(_instanzen(21))[::3]:
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        assert lo.kosten == pytest.approx(_lp(inst.a, inst.b, inst.kosten, "highs-ipm"), rel=1e-7, abs=1e-9)


def test_ganzzahlige_instanzen_gleich_networkx_und_ortools_mit_abweichung_0():
    nx = pytest.importorskip("networkx")
    pywraplp = pytest.importorskip("ortools.graph.python.min_cost_flow")
    for k in range(25):
        g = SC.instanz_rng(5, 6, k)
        n, m = 2 + g.below(5), 2 + g.below(5)
        a = [g.below(6) + 1 for _ in range(n)]
        gesamt = sum(a)
        schnitte = sorted(g.below(gesamt + 1) for _ in range(m - 1))
        b = [y - x for x, y in zip([0] + schnitte, schnitte + [gesamt])]
        Cm = np.array([[g.below(20) for _ in range(m)] for _ in range(n)], float)
        lo = S.loese_transport(a, b, Cm, "erste")
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(("L", i), demand=-a[i])
        for j in range(m):
            G.add_node(("K", j), demand=b[j])
        for i in range(n):
            for j in range(m):
                G.add_edge(("L", i), ("K", j), weight=int(Cm[i, j]))
        wert_nx, _ = nx.network_simplex(G)
        mcf = pywraplp.SimpleMinCostFlow()
        for i in range(n):
            for j in range(m):
                mcf.add_arc_with_capacity_and_unit_cost(i, n + j, gesamt, int(Cm[i, j]))
        for i in range(n):
            mcf.set_node_supply(i, a[i])
        for j in range(m):
            mcf.set_node_supply(n + j, -b[j])
        assert mcf.solve() == mcf.OPTIMAL
        assert lo.kosten == wert_nx == mcf.optimal_cost(), (k, n, m)


def test_gleiche_massen_gleich_linear_sum_assignment_und_ungarische_methode():
    lsa = scipy_opt.linear_sum_assignment
    for k in range(60):
        fam = C.FAMILIEN[k % len(C.FAMILIEN)]
        n = 3 + k % 9
        inst = SC.make_instanz(fam, n, n, 0.0, 9, 8, k)
        r, c = lsa(inst.kosten)
        referenz = inst.kosten[r, c].sum()
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "erste")
        assert lo.kosten * n == pytest.approx(referenz, abs=1e-9)
        assert ungarisch(inst.kosten)[1] == pytest.approx(referenz, abs=1e-9)


def test_dualitaet_gegen_unabhaengig_geloestes_dual_lp_und_c_transformierte():
    for inst in _instanzen(40, n_bis=15, start=3):
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        n, m = inst.kosten.shape
        # Dual-LP: max a.u + b.v unter u_i + v_j <= c_ij, u und v frei (als Minimierung mit negativem Ziel)
        A = np.zeros((n * m, n + m))
        for i in range(n):
            for j in range(m):
                A[i * m + j, i] = 1.0
                A[i * m + j, n + j] = 1.0
        r = scipy_opt.linprog(-np.concatenate([inst.a, inst.b]), A_ub=A, b_ub=inst.kosten.ravel(), bounds=(None, None), method="highs")
        assert r.status == 0
        assert -r.fun == pytest.approx(lo.kosten, rel=1e-9, abs=1e-12)                                  # starke Dualität: primal = dual (unabhängig gelöst)
        assert (lo.u[:, None] + lo.v[None, :] <= inst.kosten + 1e-9).all()                              # die Preise der Demo sind zulässig
        assert inst.a @ lo.u + inst.b @ lo.v == pytest.approx(lo.kosten, rel=1e-12, abs=1e-12)          # und haben den Dualwert
        assert ((inst.kosten - lo.u[:, None]).min(axis=0) == pytest.approx(lo.v, abs=1e-9))              # c-Transformierte v_j = min_i (c_ij - u_i)
        schlupf = inst.kosten - lo.u[:, None] - lo.v[None, :]
        assert np.abs(schlupf[lo.plan > 1e-12]).max(initial=0.0) <= 1e-9                                 # komplementärer Schlupf


def test_eindeutigkeitstest_gegen_aufzaehlung_aller_120_permutationen():
    perms = list(itertools.permutations(range(5)))
    uebereinstimmung = mehrere = 0
    for k in range(300):
        inst = SC.make_instanz("ganzzahlig 0 bis 2", 5, 5, 0.0, 77, 9, k)
        werte = np.array([inst.kosten[np.arange(5), list(p)].sum() for p in perms])
        optimal = int((werte == werte.min()).sum())
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "erste")
        uebereinstimmung += ist_eindeutig(inst.kosten, lo.plan, lo.u, lo.v) == (optimal == 1)
        mehrere += optimal > 1
    assert uebereinstimmung == 300 and mehrere >= 150


def test_eindeutigkeitstest_gegen_zweites_linearprogramm_auf_der_optimalen_flaeche():
    """Zweiter Weg: auf der Menge E0 der Zellen mit reduzierten Kosten 0 gibt es genau dann eine Richtung (Randsummen 0), die eine Zelle ausserhalb des Trägers erhöht und
    nur Zellen mit Menge größer null senkt, wenn der optimale Plan nicht eindeutig ist (Linearprogramm mit linprog)."""
    ueber_lp = nicht_eindeutig = 0
    for k in range(60):
        inst = SC.make_instanz(("Gitter L1", "ganzzahlig 0 bis 10", "ganzzahlig 0 bis 2")[k % 3], 6, 6 + k % 2, (0.0, 1.0)[k % 2], 3, 4, k)
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "letzte")
        n, m = inst.kosten.shape
        rot = inst.kosten - lo.u[:, None] - lo.v[None, :]
        e0 = (rot <= 1e-9) | (lo.plan > 1e-12)
        aus = e0 & ~(lo.plan > 1e-12)
        if not aus.any():
            assert ist_eindeutig(inst.kosten, lo.plan, lo.u, lo.v)
            continue
        idx = np.flatnonzero(e0.ravel())
        A = np.zeros((n + m, len(idx)))
        for pos, f in enumerate(idx):
            A[f // m, pos] = 1.0
            A[n + f % m, pos] = 1.0
        ziel = np.where(aus.ravel()[idx], -1.0, 0.0)
        grenzen = [(0, 1) if aus.ravel()[f] else (-1, 1) for f in idx]
        r = scipy_opt.linprog(ziel, A_eq=A, b_eq=np.zeros(n + m), bounds=grenzen, method="highs-ds")
        assert r.status == 0
        assert ist_eindeutig(inst.kosten, lo.plan, lo.u, lo.v) == (-r.fun < 1e-9), k
        ueber_lp += 1
        nicht_eindeutig += -r.fun >= 1e-9
    assert ueber_lp >= 40 and nicht_eindeutig >= 10      # der Vergleich läuft über beide Antworten, nicht nur über „eindeutig“


def test_welche_ecke_bei_gleichstand_bleibt_offen_aber_jede_gehoert_zur_optimalen_flaeche():
    inst = SC.make_instanz("Gitter L1", 10, 10, 0.0, 7, 4, 0)
    opt = _lp(inst.a, inst.b, inst.kosten)
    plaene = []
    for regel in S.REGELN:
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, regel, SC.instanz_rng(7, 4, 99))
        assert lo.kosten == pytest.approx(opt, rel=1e-9)
        assert np.abs(lo.plan.sum(axis=1) - inst.a).max() < 1e-12 and np.abs(lo.plan.sum(axis=0) - inst.b).max() < 1e-12
        plaene.append(lo.plan)
    for p, q in itertools.combinations(plaene, 2):               # jede Mischung zweier optimaler Pläne ist zulässig und ebenso gut
        mitte = 0.5 * p + 0.5 * q
        assert (inst.kosten * mitte).sum() == pytest.approx(opt, rel=1e-9)
