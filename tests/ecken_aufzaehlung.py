"""Hilfsmodul der Tests (kein Test): alle Ecken des Transportpolytops in exakter Bruchrechnung, ohne Simplex und ohne Bibliothek.

Eine Ecke gehört zu einer Basis, und eine Basis ist ein aufspannender Baum des vollständigen bipartiten Graphen K_{n,m}; der Fluss auf dem Baum ist eindeutig und folgt
durch Abschälen von Blättern. Aufgezählt werden alle Kantenmengen mit n + m - 1 Zellen, die einen Baum bilden; Ecken sind die Flüsse ohne negative Einträge."""

import itertools
from fractions import Fraction as F


def baeume(n, m):
    """Alle aufspannenden Bäume von K_{n,m} als Tupel von Zellen (Zeile, Spalte)."""
    zellen = [(i, j) for i in range(n) for j in range(m)]
    for teil in itertools.combinations(zellen, n + m - 1):
        eltern = list(range(n + m))

        def wurzel(x):
            while eltern[x] != x:
                eltern[x] = eltern[eltern[x]]
                x = eltern[x]
            return x

        kreisfrei = True
        for i, j in teil:
            ri, rj = wurzel(i), wurzel(n + j)
            if ri == rj:
                kreisfrei = False
                break
            eltern[ri] = rj
        if kreisfrei:
            yield teil


def baum_fluss(teil, a, b):
    """Der eindeutige Fluss auf dem Baum `teil` mit Zeilensummen a und Spaltensummen b (Brüche); kann negative Einträge haben."""
    n, m = len(a), len(b)
    ra = [F(x) for x in a]
    rb = [F(x) for x in b]
    offen = set(teil)
    plan = [[F(0)] * m for _ in range(n)]
    while offen:
        fertig = False
        for i in range(n):
            zellen = [z for z in offen if z[0] == i]
            if len(zellen) == 1:
                _, j = zellen[0]
                plan[i][j] = ra[i]
                rb[j] -= ra[i]
                ra[i] = F(0)
                offen.remove(zellen[0])
                fertig = True
                break
        if fertig:
            continue
        for j in range(m):
            zellen = [z for z in offen if z[1] == j]
            if len(zellen) == 1:
                i, _ = zellen[0]
                plan[i][j] = rb[j]
                ra[i] -= rb[j]
                rb[j] = F(0)
                offen.remove(zellen[0])
                fertig = True
                break
        assert fertig
    return plan


def ecken(a, b):
    """{Plan als Tupel von Tupeln von Brüchen: Liste der Bäume, die ihn liefern} für alle zulässigen Basislösungen."""
    n, m = len(a), len(b)
    aus = {}
    for t in baeume(n, m):
        plan = baum_fluss(t, a, b)
        if all(x >= 0 for zeile in plan for x in zeile):
            aus.setdefault(tuple(tuple(z) for z in plan), []).append(t)
    return aus


def kosten(plan, C):
    return sum(plan[i][j] * C[i][j] for i in range(len(C)) for j in range(len(C[0])))


def optimale_ecken(a, b, C):
    """(Optimalwert, Liste der optimalen Ecken) in Brüchen."""
    alle = ecken(a, b)
    werte = {p: kosten(p, C) for p in alle}
    opt = min(werte.values())
    return opt, [p for p, w in werte.items() if w == opt]
