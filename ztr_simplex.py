"""Transportsimplex mit Basisbaum (nur numpy und Standardbibliothek): der Kern der Demo, in einer eigenen Datei und ohne Streamlit.

Aufgabe: minimiere die Summe c_ij x_ij unter x >= 0, Zeilensummen a (Angebote der Lager) und Spaltensummen b (Nachfragen der Kunden), Summe a = Summe b.
Verfahren (Lehrbuchform des Transportsimplex): Startecke nach der Nordwest-Ecken-Regel, Preise (Potentiale) u, v aus dem Basisbaum mit u_0 = 0,
reduzierte Kosten d_ij = c_ij - u_i - v_j, eine Nichtbasiszelle mit d_ij < 0 tritt ein, der Kreis im Baum bestimmt die austretende Zelle.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben:
  nordwest_ecke        Startbasis (ein aufspannender Baum mit n + m - 1 Zellen, bei Entartung mit Zellen der Menge 0)
  potentiale           Preise u, v aus dem Baum (und die Eltern-Zeiger, aus denen sich der Kreis ergibt)
  reduzierte_kosten    d = c - u - v
  waehle_eintritt      Eintrittsregel: erste, letzte oder zufällige Zelle mit d < 0
  kreis_zellen         die Zellen des Kreises mit Vorzeichen
  pivotiere            Mengen ändern, austretende Zelle wählen
  loese_transport      ruft nur der Reihe nach auf
  pruefe_loesung       unabhängiger Prüfer (Zulässigkeit, Zulässigkeit der Preise, komplementärer Schlupf), läuft auch in der App

Die Eintrittsregel entscheidet bei Gleichstand, WELCHE optimale Ecke herauskommt (nicht, ob der Wert stimmt). Die Regel „erste“ mit „kleinster Index
verlässt die Basis“ ist die Regel von Bland: sie kreist nach dem Lehrbuchsatz nicht. „letzte“ ist dieselbe Regel mit umgekehrter Reihenfolge der Zellen. „zufällige“ wechselt
nach ZAEHLER_BLAND entarteten Schritten in Folge zur Regel von Bland, bis ein Schritt den Wert senkt; auch dann kreist das Verfahren nach dem Lehrbuchsatz nicht.
„steilste“ (größte Abnahme der reduzierten Kosten, Dantzig) ist die vierte Regel für die schnellen Reihen der App; sie gehört nicht zu REGELN, den dreien der Auswahl.
Der Zufall der Regel „zufällige“ kommt aus einem übergebenen Generator (SplitMix64, `ztr_rng.SplitMix64`), nie aus einem globalen Zustand.

Grenzen: Summe a muss gleich Summe b sein (sonst ValueError; ungleiche Summen sind Aufgabe einer anderen Demo der Reihe). Mengen unter TOL_MENGE
gelten als null. Masse 0 bei einem Lager oder Kunden ist erlaubt (die Zelle bleibt als Menge 0 in der Basis)."""

from dataclasses import dataclass

import numpy as np

REGELN = ("erste", "letzte", "zufällige")   # die drei Regeln, die die App zur Wahl stellt
ALLE_REGELN = REGELN + ("steilste",)         # „steilste“: größte Abnahme der reduzierten Kosten (Dantzig), schnell; die Reihen der App rechnen damit
TOL_MENGE = 1e-12          # Mengen darunter gelten als null (Nichtnullen zählen braucht eine Rauschgrenze)
TOL_REL = 1e-11            # relative Toleranz der reduzierten Kosten, bezogen auf max(1, größter Betrag der Kosten)
SUMMEN_TOL = 1e-9          # erlaubte relative Abweichung von Summe a und Summe b
ZAEHLER_BLAND = 25         # so viele entartete Schritte in Folge, dann Regel von Bland bis zum nächsten echten Schritt


@dataclass(frozen=True)
class Loesung:
    plan: np.ndarray          # n x m, Mengen unter TOL_MENGE auf 0 gesetzt
    kosten: float
    u: np.ndarray             # Preise der Lager, u_0 = 0 (die Preise gelten nur bis auf eine Konstante)
    v: np.ndarray             # Preise der Kunden
    basis: np.ndarray         # bool, n x m, genau n + m - 1 Zellen (ein aufspannender Baum)
    schritte: int             # Zahl der Pivotschritte
    entartete_schritte: int   # davon Schritte mit Schrittweite 0
    bland_schritte: int       # davon Schritte nach der Regel von Bland (Rückfall der Regel „zufällige“)


def toleranz(C) -> float:
    return TOL_REL * max(1.0, float(np.abs(C).max(initial=0.0)))


def _pruefe_eingabe(a, b, C):
    a, b, C = np.asarray(a, float), np.asarray(b, float), np.asarray(C, float)
    if a.ndim != 1 or b.ndim != 1 or C.shape != (len(a), len(b)):
        raise ValueError("a (n), b (m) und C (n x m) müssen zueinander passen")
    if len(a) == 0 or len(b) == 0:
        raise ValueError("mindestens ein Lager und ein Kunde")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("Mengen dürfen nicht negativ sein")
    if abs(a.sum() - b.sum()) > SUMMEN_TOL * max(1.0, a.sum()):
        raise ValueError(f"Summe der Angebote ({a.sum():.6g}) und der Nachfragen ({b.sum():.6g}) sind verschieden")
    return a, b, C


def nordwest_ecke(a, b):
    """Startecke: von links oben die Mengen füllen. Gibt (x, basis) zurück; basis hat n + m - 1 Zellen, auch wenn Zeile und Spalte gleichzeitig leer werden
    (dann folgt eine Zelle mit Menge 0, damit die Basis ein Baum bleibt)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    x = np.zeros((n, m))
    basis = np.zeros((n, m), dtype=bool)
    ra, rb = a.copy(), b.copy()
    i = j = 0
    while True:
        t = min(ra[i], rb[j])
        x[i, j] = t
        basis[i, j] = True
        ra[i] -= t
        rb[j] -= t
        if i == n - 1 and j == m - 1:
            break
        if i == n - 1:
            j += 1
        elif j == m - 1:
            i += 1
        elif ra[i] == 0.0:
            i += 1
        else:
            j += 1
    return x, basis


def _nachbarn(basis):
    n, m = basis.shape
    zeilen = [[] for _ in range(n)]
    spalten = [[] for _ in range(m)]
    for i, j in zip(*np.nonzero(basis)):
        zeilen[i].append(int(j))
        spalten[j].append(int(i))
    return zeilen, spalten


def potentiale(basis, C):
    """Preise u, v mit u_i + v_j = c_ij auf allen Basiszellen und u_0 = 0. Gibt (u, v, eltern, tiefe) zurück; Knoten 0..n-1 sind die Lager, n..n+m-1 die Kunden;
    `eltern` und `tiefe` beschreiben den Baum von Lager 0 aus (für den Kreis)."""
    n, m = basis.shape
    zeilen, spalten = _nachbarn(basis)
    u = np.zeros(n)
    v = np.zeros(m)
    eltern = [-1] * (n + m)
    tiefe = [-1] * (n + m)
    tiefe[0] = 0
    schlange = [0]
    k = 0
    while k < len(schlange):
        knoten = schlange[k]
        k += 1
        if knoten < n:
            for j in zeilen[knoten]:
                if tiefe[n + j] < 0:
                    v[j] = C[knoten, j] - u[knoten]
                    eltern[n + j] = knoten
                    tiefe[n + j] = tiefe[knoten] + 1
                    schlange.append(n + j)
        else:
            j = knoten - n
            for i in spalten[j]:
                if tiefe[i] < 0:
                    u[i] = C[i, j] - v[j]
                    eltern[i] = knoten
                    tiefe[i] = tiefe[knoten] + 1
                    schlange.append(i)
    if len(schlange) != n + m:
        raise RuntimeError("die Basis ist kein zusammenhängender Baum")
    return u, v, eltern, tiefe


def reduzierte_kosten(C, u, v):
    return np.asarray(C, float) - u[:, None] - v[None, :]


def waehle_eintritt(d, basis, regel, tol, rng=None):
    """Eintrittsregel: eine Nichtbasiszelle mit d < -tol (flacher Index in Zeilenfolge) oder None, wenn keine existiert (optimal)."""
    kandidaten = np.flatnonzero(((d < -tol) & ~basis).ravel())
    if kandidaten.size == 0:
        return None
    if regel == "erste":
        return int(kandidaten[0])
    if regel == "letzte":
        return int(kandidaten[-1])
    if regel == "zufällige":
        if rng is None:
            raise ValueError("die Regel „zufällige“ braucht einen Generator")
        return int(kandidaten[rng.below(int(kandidaten.size))])
    if regel == "steilste":
        return int(kandidaten[np.argmin(d.ravel()[kandidaten])])   # bei Gleichstand der kleinste Index
    raise ValueError(f"unbekannte Eintrittsregel: {regel}")


def kreis_zellen(n, eintritt, eltern, tiefe, m):
    """Kreis des Eintritts (i, j) im Baum: Liste ((Zeile, Spalte), Vorzeichen) mit +1 für die Eintrittszelle und abwechselnd -1, +1, ... entlang des Baumweges
    von Kunde j zu Lager i."""
    i, j = divmod(eintritt, m)
    links = [n + j]
    rechts = [i]
    a_, b_ = n + j, i
    while a_ != b_:
        if tiefe[a_] >= tiefe[b_]:
            a_ = eltern[a_]
            links.append(a_)
        else:
            b_ = eltern[b_]
            rechts.append(b_)
    weg = links + rechts[-2::-1]   # beide Listen enden im gemeinsamen Vorfahren; er steht nur einmal im Weg
    zellen = [((i, j), +1)]
    for k in range(len(weg) - 1):
        p, q = weg[k], weg[k + 1]
        zeile, spalte = (q, p - n) if p >= n else (p, q - n)
        zellen.append(((zeile, spalte), -1 if k % 2 == 0 else +1))
    return zellen


def pivotiere(x, basis, kreis, regel, m):
    """Schrittweite = kleinste Menge auf den Minus-Zellen; die austretende Zelle ist bei Gleichstand die mit kleinstem (Regel „letzte“: größtem) flachen Index.
    Ändert x und basis (an Ort und Stelle). Gibt (Schrittweite, austretender flacher Index) zurück."""
    minus = [(z[0] * m + z[1], z) for z, s in kreis if s < 0]
    theta = min(x[z] for _, z in minus)
    gleich = [(idx, z) for idx, z in minus if x[z] <= theta + TOL_MENGE]
    idx_aus, zelle_aus = (max if regel == "letzte" else min)(gleich, key=lambda t: t[0])
    for z, s in kreis:
        x[z] += s * theta
    for _, z in minus:
        if x[z] < TOL_MENGE:
            if x[z] < -TOL_MENGE:
                raise RuntimeError("negative Menge im Pivotschritt")
            x[z] = 0.0
    x[zelle_aus] = 0.0
    eintritt = kreis[0][0]
    basis[eintritt] = True
    basis[zelle_aus] = False
    return float(theta), idx_aus


def loese_transport(a, b, C, regel="erste", rng=None, max_schritte=None):
    """Löst das Transportproblem mit dem Basisbaum-Simplex und gibt eine Ecke des Transportpolytops zurück (siehe Kopf). `rng` nur für „zufällige“."""
    if regel not in ALLE_REGELN:
        raise ValueError(f"unbekannte Eintrittsregel: {regel}")
    a, b, C = _pruefe_eingabe(a, b, C)
    n, m = C.shape
    b = b * (a.sum() / b.sum()) if b.sum() > 0 else b
    tol = toleranz(C)
    x, basis = nordwest_ecke(a, b)
    grenze = max_schritte if max_schritte is not None else 20 * n * m + 1000
    schritte = entartet = bland = folge = 0
    while True:
        u, v, eltern, tiefe = potentiale(basis, C)
        d = reduzierte_kosten(C, u, v)
        modus = "erste" if (regel in ("zufällige", "steilste") and folge >= ZAEHLER_BLAND) else regel
        eintritt = waehle_eintritt(d, basis, modus, tol, rng)
        if eintritt is None:
            break
        if schritte >= grenze:
            raise RuntimeError(f"keine Konvergenz nach {schritte} Schritten")
        kreis = kreis_zellen(n, eintritt, eltern, tiefe, m)
        theta, _ = pivotiere(x, basis, kreis, modus, m)
        schritte += 1
        if modus != regel:
            bland += 1
        if theta <= TOL_MENGE:
            entartet += 1
            folge += 1
        else:
            folge = 0
    plan = np.where(x < TOL_MENGE, 0.0, x)
    return Loesung(plan, float((C * plan).sum()), u, v, basis, schritte, entartet, bland)


def pruefe_loesung(a, b, C, loesung, tol_summe=1e-9):
    """Unabhängiger Prüfer ohne Verfahrenslogik: Liste der verletzten Bedingungen (leer = in Ordnung).
    Zulässig (Randsummen, nicht negativ), Preise zulässig (u_i + v_j <= c_ij), komplementärer Schlupf (Plan nur auf Zellen mit d = 0), Primal = Dual."""
    a, b, C = np.asarray(a, float), np.asarray(b, float), np.asarray(C, float)
    plan = loesung.plan
    fehler = []
    skala = max(1.0, a.sum())
    if (plan < -TOL_MENGE).any():
        fehler.append("negative Menge im Plan")
    if np.abs(plan.sum(axis=1) - a).max(initial=0.0) > tol_summe * skala:
        fehler.append("Zeilensummen weichen von den Angeboten ab")
    if np.abs(plan.sum(axis=0) - b).max(initial=0.0) > tol_summe * skala:
        fehler.append("Spaltensummen weichen von den Nachfragen ab")
    d = reduzierte_kosten(C, loesung.u, loesung.v)
    tol = toleranz(C)
    if (d < -tol).any():
        fehler.append("Preise verletzen u_i + v_j <= c_ij")
    if (np.abs(d)[plan > TOL_MENGE] > tol * 1e3).any():
        fehler.append("Plan liegt auf einer Zelle mit positiven reduzierten Kosten")
    dual = float(a @ loesung.u + b @ loesung.v)
    if abs(dual - float((C * plan).sum())) > 1e-8 * max(1.0, abs(dual)):
        fehler.append("Primalwert und Dualwert weichen ab")
    return fehler
