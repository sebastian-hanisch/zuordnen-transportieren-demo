"""Eindeutigkeitstest für den optimalen Plan (nur numpy, ohne Linearoptimierungs-Bibliothek, ohne Streamlit).

Idee: Jeder optimale Plan liegt auf den Zellen mit reduzierten Kosten 0 (komplementärer Schlupf mit den optimalen Preisen u, v), und jeder zulässige Plan
auf diesen Zellen ist optimal. Die optimale Menge ist also die Menge der Transportpläne auf der Zellmenge E0 = {d_ij = 0}. Sie besteht aus einem einzigen
Punkt genau dann, wenn es von der gefundenen Ecke aus keine zulässige Richtung gibt, also keinen Kreis von Verschiebungen auf E0, der nur Zellen mit Menge
größer null verkleinert. Als gerichteter Graph über den Knoten „Lager i“ und „Kunde j“: für jede Zelle in E0 ein Pfeil Lager -> Kunde (Menge dort erhöhen),
für jede Zelle in E0 mit Menge größer null zusätzlich der Rückpfeil Kunde -> Lager (Menge dort senken). Ein Kreis, der nicht nur eine Zelle hin und zurück
läuft, ist eine zulässige Richtung. Es gibt keinen solchen Kreis genau dann, wenn in jeder starken Zusammenhangskomponente alle Pfeile Hin-und-Rück-Paare
sind und diese Paare einen Baum bilden (Zellenzahl = Knotenzahl - 1).

Die Reihenfolge der Ausgabe ist fest; der Test braucht Preise u, v zu einer optimalen Ecke (aus `ztr_simplex.loese_transport`)."""

import numpy as np

from ztr_simplex import TOL_MENGE


def _erreichbar(pfeile):
    """Boolesche Erreichbarkeitsmatrix (transitive Hülle, wiederholte Quadrierung)."""
    r = pfeile.copy()
    np.fill_diagonal(r, True)
    while True:
        neu = (r.astype(np.int32) @ r.astype(np.int32)) > 0
        if (neu == r).all():
            return r
        r = neu


def null_zellen(C, u, v, plan, tol_rel=1e-9):
    """E0: Zellen mit reduzierten Kosten höchstens tol (bezogen auf max(1, größter Kostenbetrag)); Zellen mit Menge größer null gehören immer dazu."""
    C = np.asarray(C, float)
    tol = tol_rel * max(1.0, float(np.abs(C).max(initial=0.0)))
    return ((C - u[:, None] - v[None, :]) <= tol) | (np.asarray(plan) > TOL_MENGE)


def ist_eindeutig(C, plan, u, v, tol_rel=1e-9):
    """True, wenn der optimale Plan der einzige ist (siehe Kopf). `plan`, `u`, `v` gehören zu einer optimalen Ecke."""
    plan = np.asarray(plan, float)
    n, m = plan.shape
    e0 = null_zellen(C, u, v, plan, tol_rel)
    traeger = plan > TOL_MENGE
    pfeile = np.zeros((n + m, n + m), dtype=bool)
    pfeile[:n, n:] = e0
    pfeile[n:, :n] = (e0 & traeger).T
    r = _erreichbar(pfeile)
    stark = r & r.T                                   # derselbe starke Zusammenhang
    for i in range(n):
        for j in range(m):
            if not e0[i, j]:
                continue
            if stark[i, n + j] and not traeger[i, j]:  # Hinpfeil ohne Rückpfeil innerhalb einer Komponente: es gibt einen echten Kreis
                return False
    # Alle Pfeile innerhalb der Komponenten sind Hin-und-Rück-Paare; die Paare müssen Bäume bilden
    gesehen = np.zeros(n + m, dtype=bool)
    for start in range(n + m):
        if gesehen[start]:
            continue
        knoten = np.flatnonzero(stark[start])
        gesehen[knoten] = True
        if len(knoten) == 1:
            continue
        zellen = sum(1 for k in knoten if k < n for l in knoten if l >= n and e0[k, l - n] and traeger[k, l - n])
        if zellen != len(knoten) - 1:
            return False
    return True
