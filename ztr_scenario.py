"""Szenario: Instanzen des Transportproblems (Lager mit Angeboten, Kunden mit Nachfragen, Kostenmatrix) aus dem Hausgenerator SplitMix64 (`ztr_rng.py`).

Alles ist deterministisch aus (Zufalls-Seed, Zweck, Index): gleiche Eingabe, gleiche Instanz, auf jeder Plattform. Masse: Gesamtmenge 1 auf beiden Seiten.
Massen a = (1 - s) / n + s · Dirichlet(1) (Dirichlet(1) = n unabhängige Exponentialzahlen, durch ihre Summe geteilt), b ebenso; s = 0 gibt gleiche Massen 1/n.
Kostenfamilien: siehe `ztr_constants.FAMILIEN`. Neben den Kosten C hat jede Instanz eine zweite Kostenmatrix D: ein zweites Kriterium neben den Kosten, mit dem
Experiment 3 zeigt, dass gleich teure Pläne nach einem anderen Maß verschieden sein können (Gitter: das andere Abstandsmaß, sonst eine unabhängig gezogene zweite Matrix)."""

import math
from dataclasses import dataclass

import numpy as np

import ztr_constants as C
from ztr_rng import MASK, SplitMix64


@dataclass(frozen=True)
class Instanz:
    a: np.ndarray                       # Angebote der n Lager (Summe 1, bei Handbeispielen die ganzen Mengen)
    b: np.ndarray                       # Nachfragen der m Kunden
    kosten: np.ndarray                  # n x m
    zweit: np.ndarray                   # n x m, zweites Kriterium (siehe Kopf)
    lager: np.ndarray | None = None     # n x 2 Orte (nur bei geometrischen Familien)
    kunden: np.ndarray | None = None    # m x 2 Orte
    familie: str = ""
    gitter_seite: int = 0               # tatsächlich benutzte Seitenlänge (nur Gitter), sonst 0
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.a)

    @property
    def m(self) -> int:
        return len(self.b)


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Seed, Zweck, Index); die ersten Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def massen(g: SplitMix64, n: int, s: float) -> np.ndarray:
    """Angebote bzw. Nachfragen mit Summe 1: (1 - s)/n + s · Dirichlet(1)."""
    if s == 0.0:
        return np.full(n, 1.0 / n)
    e = np.array([g.expovariate(1.0) for _ in range(n)])
    return (1.0 - s) / n + s * e / e.sum()


def gitter_seite(n: int, m: int, gewuenscht: int, ziehung: str) -> int:
    """Seitenlänge des Gitters. Standard (0): ceil(sqrt(n + m)) + 1. Ohne Zurücklegen müssen n + m verschiedene Punkte Platz haben (Seite^2 >= n + m)."""
    standard = math.ceil(math.sqrt(n + m)) + 1
    seite = standard if gewuenscht == C.GITTER_STANDARD else int(gewuenscht)
    if ziehung == C.ZIEHUNGEN[0]:
        seite = max(seite, math.ceil(math.sqrt(n + m)))
    return seite


def _gitterpunkte(g: SplitMix64, anzahl: int, seite: int, ziehung: str) -> np.ndarray:
    """`anzahl` Gitterpunkte (x, y) mit 0 <= x, y < seite; ohne Zurücklegen alle verschieden (teilweises Mischen), sonst unabhängig."""
    if ziehung == C.ZIEHUNGEN[0]:
        felder = list(range(seite * seite))
        for k in range(anzahl):
            j = k + g.below(len(felder) - k)
            felder[k], felder[j] = felder[j], felder[k]
        gewaehlt = felder[:anzahl]
    else:
        gewaehlt = [g.below(seite * seite) for _ in range(anzahl)]
    return np.array([(f // seite, f % seite) for f in gewaehlt], dtype=float)


def _abstaende(X: np.ndarray, Y: np.ndarray):
    delta = X[:, None, :] - Y[None, :, :]
    return (delta ** 2).sum(-1), np.abs(delta).sum(-1)      # quadrierter euklidischer Abstand, L1


def _zufallsmatrix(g: SplitMix64, n: int, m: int, familie: str) -> np.ndarray:
    if familie == "stetig (Zufallskosten)":
        return np.array([[g.random() for _ in range(m)] for _ in range(n)])
    hoechst = {"ganzzahlig 0 bis 100": 100, "ganzzahlig 0 bis 10": 10, "ganzzahlig 0 bis 2": 2}[familie]
    return np.array([[float(g.below(hoechst + 1)) for _ in range(m)] for _ in range(n)])


def make_instanz(familie: str, n: int, m: int, s: float, seed: int, zweck: int, index: int,
                 gitter: int = C.GITTER_STANDARD, ziehung: str = C.ZIEHUNGEN[0]) -> Instanz:
    """Zufallsinstanz mit n Lagern und m Kunden. Reihenfolge der Ziehungen: Massen a, Massen b, Orte bzw. Kostenmatrix, zweite Kostenmatrix."""
    if familie not in C.FAMILIEN:
        raise ValueError(f"unbekannte Kostenfamilie: {familie}")
    if n < 1 or m < 1:
        raise ValueError("mindestens ein Lager und ein Kunde")
    g = instanz_rng(seed, zweck, index)
    a = massen(g, n, s)
    b = massen(g, m, s)
    if familie == "Punkte im Einheitsquadrat":
        X = np.array([[g.random(), g.random()] for _ in range(n)])
        Y = np.array([[g.random(), g.random()] for _ in range(m)])
        q, l1 = _abstaende(X, Y)
        return Instanz(a, b, q / q.max(), l1 / l1.max(), X, Y, familie)
    if familie in C.GITTER_FAMILIEN:
        seite = gitter_seite(n, m, gitter, ziehung)
        punkte = _gitterpunkte(g, n + m, seite, ziehung)
        X, Y = punkte[:n], punkte[n:]
        q, l1 = _abstaende(X, Y)
        kosten, zweit = (l1, q) if familie == "Gitter L1" else (q, l1)
        return Instanz(a, b, kosten, zweit, X, Y, familie, seite)
    kosten = _zufallsmatrix(g, n, m, familie)
    return Instanz(a, b, kosten, _zufallsmatrix(g, n, m, familie), None, None, familie)


def handbeispiel_zuordnung() -> Instanz:
    """Handbeispiel 1: drei Lager, drei Kunden, Massen 1; Kosten Zeilen (4, 1, 3), (2, 0, 5), (3, 2, 2). Es hat keine Orte."""
    kosten = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    return Instanz(np.ones(3), np.ones(3), kosten, kosten, None, None, "Handbeispiel", 0, C.QUELLE_HAND_ZUORDNUNG)


HAND_LAGER = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]])
HAND_KUNDEN = np.array([[0.0, 2.0], [4.0, 2.0], [2.0, 5.0]])


def handbeispiel_transport(a=(9.0, 6.0, 5.0), b=(8.0, 8.0, 4.0)) -> Instanz:
    """Handbeispiel 2: Lager (0,0), (4,0), (2,3) mit Angeboten 9, 6, 5; Kunden (0,2), (4,2), (2,5) mit Nachfragen 8, 8, 4; Kosten = quadrierter Abstand."""
    q, l1 = _abstaende(HAND_LAGER, HAND_KUNDEN)
    return Instanz(np.array(a, float), np.array(b, float), q, l1, HAND_LAGER, HAND_KUNDEN, "Handbeispiel", 0, C.QUELLE_HAND_TRANSPORT)


def instanz_aus_einstellungen(einstellung) -> Instanz:
    """Die Instanz der Ansicht oben: Handbeispiel oder Zufallsinstanz (Index 0 des Zwecks ANSICHT)."""
    if einstellung.quelle == C.QUELLE_HAND_ZUORDNUNG:
        return handbeispiel_zuordnung()
    if einstellung.quelle == C.QUELLE_HAND_TRANSPORT:
        return handbeispiel_transport()
    return make_instanz(einstellung.familie, einstellung.n, einstellung.n, einstellung.s, einstellung.seed, C.ZWECK_ANSICHT, 0,
                        einstellung.gitter, einstellung.ziehung)
