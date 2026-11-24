"""Fünf Fremdlöser auf denselben Zuordnungsinstanzen (n = 10, gleiche Massen): liefern sie bei gleichem Wert verschiedene Pläne?

Aufruf aus dem Demo-Ordner mit dem Test-venv (SciPy, NetworkX, OR-Tools): `python tools/messreihen_fremdloeser.py`; schreibt `tools/messreihen_fremdloeser.json`.
Die Fremdlöser gehören nicht zur Laufzeit der Demo (nur Gegenproben der Tests); die Zahl gilt für diese Löser in diesen Versionen (die Datei nennt sie):
`scipy.optimize.linprog` mit HiGHS-Simplex (`highs-ds`) und HiGHS-Innere-Punkte (`highs-ipm`, mit Crossover), `scipy.optimize.linear_sum_assignment`,
der Mindestkostenfluss von OR-Tools und `networkx.network_simplex`. Zähleinheit: die Instanz; „verschieden“ heißt, dass sich mindestens zwei der fünf Pläne
in einem Eintrag um mehr als 1e-6 unterscheiden. Welche Ecke ein Löser bei Gleichstand wählt, kann sich zwischen Versionen und Plattformen ändern."""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import ortools
import scipy
from ortools.graph.python import min_cost_flow
from scipy.optimize import linear_sum_assignment, linprog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ztr_constants as C  # noqa: E402
import ztr_scenario as SC  # noqa: E402

MESS_SEED = 2026
FAMILIEN = (C.FAMILIEN[1], C.FAMILIEN[2], C.FAMILIEN[3], C.FAMILIEN[4], "Gitter L1", "Gitter L2²")


def plaene_der_fuenf_loeser(kosten: np.ndarray) -> dict:
    """Die Pläne (n x n, 0/1) der fünf Löser auf einer Zuordnung mit Massen 1."""
    n = kosten.shape[0]
    A = np.vstack([np.kron(np.eye(n), np.ones((1, n))), np.kron(np.ones((1, n)), np.eye(n))])
    aus = {}
    for name in ("highs-ds", "highs-ipm"):
        r = linprog(kosten.ravel(), A_eq=A, b_eq=np.ones(2 * n), bounds=(0, None), method=name)
        assert r.status == 0, r.message
        aus[name] = np.round(r.x.reshape(n, n), 6)
    zeilen, spalten = linear_sum_assignment(kosten)
    p = np.zeros((n, n))
    p[zeilen, spalten] = 1.0
    aus["linear_sum_assignment"] = p
    skala = 10 ** 9
    mcf = min_cost_flow.SimpleMinCostFlow()
    for i in range(n):
        for j in range(n):
            mcf.add_arc_with_capacity_and_unit_cost(i, n + j, 1, int(round(kosten[i, j] * skala)))
    for i in range(n):
        mcf.set_node_supply(i, 1)
        mcf.set_node_supply(n + i, -1)
    assert mcf.solve() == mcf.OPTIMAL
    p = np.zeros((n, n))
    for arc in range(mcf.num_arcs()):
        if mcf.flow(arc) > 0:
            p[mcf.tail(arc), mcf.head(arc) - n] = 1.0
    aus["ortools"] = p
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(("L", i), demand=-1)
        G.add_node(("K", i), demand=1)
        for j in range(n):
            G.add_edge(("L", i), ("K", j), weight=int(round(kosten[i, j] * skala)), capacity=1)
    _, fluss = nx.network_simplex(G)
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p[i, j] = fluss[("L", i)][("K", j)]
    aus["networkx"] = p
    return aus


def reihe(familie: str, n: int, instanzen: int) -> dict:
    verschieden = 0
    gleicher_wert = 0
    for i in range(instanzen):
        inst = SC.make_instanz(familie, n, n, 0.0, MESS_SEED + 7, C.ZWECK_GLEICHSTAND, i)
        plaene = plaene_der_fuenf_loeser(inst.kosten)
        werte = [float((inst.kosten * p).sum()) for p in plaene.values()]
        gleicher_wert += max(werte) - min(werte) <= 1e-6 * max(1.0, abs(max(werte)))
        erste = next(iter(plaene.values()))
        verschieden += any(np.abs(p - erste).max() > 1e-6 for p in plaene.values())
    return {"familie": familie, "n": n, "instanzen": instanzen, "verschiedene_plaene": int(verschieden), "gleicher_wert": int(gleicher_wert)}


def main() -> None:
    aus = {"meta": {"seed": MESS_SEED + 7, "scipy": scipy.__version__, "networkx": nx.__version__, "ortools": ortools.__version__, "numpy": np.__version__,
                    "loeser": ["linprog highs-ds", "linprog highs-ipm", "linear_sum_assignment", "ortools SimpleMinCostFlow", "networkx network_simplex"]},
           "reihen": [reihe(f, 10, 150) for f in FAMILIEN]}
    (ROOT / "tools" / "messreihen_fremdloeser.json").write_text(json.dumps(aus, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for r in aus["reihen"]:
        print(r)


if __name__ == "__main__":
    main()
