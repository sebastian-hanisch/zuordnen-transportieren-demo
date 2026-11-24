"""Ungarische Methode (Kuhn-Munkres in der Fassung mit kürzesten erweiternden Wegen und Potentialen), nur numpy, ohne Streamlit.

Löst die Zuordnung: jedem der n Lager genau einen der n Kunden, so dass die Summe der Kosten kleinstmöglich ist (Massen alle gleich, ganze Paare). Sie dient hier
als Vergleich für den Transportsimplex bei gleichen Massen 1/n: bei gleichen Massen sind beide Aufgaben dieselbe, und der Wert muss übereinstimmen.
Bei Gleichstand wählt sie unter den optimalen Zuordnungen nach ihrer Zeilenreihenfolge; welche, ist keine Eigenschaft der Aufgabe."""

import numpy as np


def ungarisch(C):
    """Gibt (zuordnung, kosten) zurück: zuordnung[i] = Spalte der Zeile i. C muss quadratisch sein."""
    C = np.asarray(C, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] == 0:
        raise ValueError("die Ungarische Methode braucht eine quadratische, nichtleere Kostenmatrix")
    n = C.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)      # p[j] = Zeile (1-basiert), die Spalte j gehört; 0 = frei
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        verwendet = np.zeros(n + 1, dtype=bool)
        weg = np.zeros(n + 1, dtype=int)
        while True:
            verwendet[j0] = True
            i0 = p[j0]
            cur = C[i0 - 1] - u[i0] - v[1:]
            frei = ~verwendet[1:]
            besser = frei & (cur < minv[1:])
            minv[1:][besser] = cur[besser]
            weg[1:][besser] = j0
            kandidat = np.where(frei, minv[1:], np.inf)
            j1 = int(np.argmin(kandidat)) + 1
            delta = float(kandidat[j1 - 1])
            u[p[verwendet]] += delta
            v[verwendet] -= delta
            minv[~verwendet] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = weg[j0]
            p[j0] = p[j1]
            j0 = j1
    zuordnung = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        zuordnung[p[j] - 1] = j - 1
    return zuordnung, float(C[np.arange(n), zuordnung].sum())
