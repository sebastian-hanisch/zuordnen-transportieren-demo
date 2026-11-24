"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte.

Rechenaufwand am Regler-Maximum (Komplexität des Transportsimplex je Instanz: etwa O(Schritte · n · m) mit 150 bis 1 000 Schritten bei n = m = 40):
Die Reihen der App laufen je Experiment über INSTANZEN_MAX Instanzen; bei n = 40 dauert eine Reihe von 200 Instanzen etwa 5 bis 120 Sekunden je nach Experiment und Rechner (gemessen auf einem Windows-Rechner unter paralleler Last: 6 s, 26 s und 121 s) (Spinner, Zwischenspeicher)."""

FAMILIEN = (
    "Punkte im Einheitsquadrat",
    "stetig (Zufallskosten)",
    "ganzzahlig 0 bis 100",
    "ganzzahlig 0 bis 10",
    "ganzzahlig 0 bis 2",
    "Gitter L1",
    "Gitter L2²",
)
FAMILIEN_HILFE = {
    "Punkte im Einheitsquadrat": "Lager und Kunden sind Zufallspunkte im Einheitsquadrat, die Kosten der quadrierte Abstand geteilt durch den größten Wert (höchstens 1).",
    "stetig (Zufallskosten)": "Jede Zelle der Kostenmatrix ist unabhängig gleichverteilt zwischen 0 und 1; keine Geometrie.",
    "ganzzahlig 0 bis 100": "Jede Zelle ist eine unabhängig gezogene ganze Zahl von 0 bis 100; keine Geometrie.",
    "ganzzahlig 0 bis 10": "Jede Zelle ist eine unabhängig gezogene ganze Zahl von 0 bis 10; keine Geometrie, viele Gleichstände.",
    "ganzzahlig 0 bis 2": "Jede Zelle ist 0, 1 oder 2; keine Geometrie, viele Gleichstände (ab n = 10 fast immer).",
    "Gitter L1": "Lager und Kunden liegen auf ganzzahligen Gitterpunkten, die Kosten sind die Summe der Beträge der Koordinatenunterschiede (L1).",
    "Gitter L2²": "Lager und Kunden liegen auf ganzzahligen Gitterpunkten, die Kosten sind der quadrierte euklidische Abstand.",
}
GITTER_FAMILIEN = ("Gitter L1", "Gitter L2²")
GEOMETRISCHE_FAMILIEN = ("Punkte im Einheitsquadrat",) + GITTER_FAMILIEN   # nur hier gibt es Orte, an denen Pfeile gezeichnet werden

QUELLE_ZUFALL = "Zufallsinstanz"
QUELLE_HAND_ZUORDNUNG = "Handbeispiel: Zuordnung 3 × 3"
QUELLE_HAND_TRANSPORT = "Handbeispiel: Transport 3 × 3"
QUELLEN = (QUELLE_HAND_ZUORDNUNG, QUELLE_HAND_TRANSPORT, QUELLE_ZUFALL)

REGELN = ("erste", "letzte", "zufällige")
REGEL_REIHEN = "steilste"          # die Reihen (Experimente 1 und 2, Anteil mehrerer Optima) rechnen mit der schnellen Regel „steilste“
S_STUFEN = (0.0, 0.001, 0.01, 0.1, 1.0)
S_MISCH = (0.0, 0.25, 0.5, 0.75, 1.0)
GITTER_STANDARD = 0                # 0 = Standard (Seitenlänge ceil(sqrt(n + m)) + 1), sonst die gewählte Seitenlänge
GITTER_MIN, GITTER_MAX = 5, 20
ZIEHUNGEN = ("ohne Zurücklegen", "mit Zurücklegen")

N_MIN, N_MAX, DEFAULT_N = 3, 40, 10         # n = m (Lager = Kunden); Komplexität je Reihe O(Instanzen · Schritte · n²)
INSTANZEN_MIN, INSTANZEN_MAX, DEFAULT_INSTANZEN, INSTANZEN_STEP = 10, 200, 100, 10
SEED_MAX, DEFAULT_SEED = 999999, 35
ZWECK_ANSICHT, ZWECK_ZUORDNEN, ZWECK_UEBERGANG, ZWECK_GLEICHSTAND = 1, 2, 3, 4   # getrennte Zufallsströme je Experiment

COLOR_LAGER = "#2F6B65"
COLOR_KUNDE = "#c0504d"
COLOR_PFEIL = "rgba(47,107,101,0.75)"
COLOR_LINE = "#4c78a8"
COLOR_BAND = "rgba(76,120,168,0.18)"

DEFAULTS = {
    "quelle": QUELLE_HAND_ZUORDNUNG,
    "familie": "Gitter L1",
    "n_slider": DEFAULT_N,
    "s_stufe": 0.0,
    "instanzen": DEFAULT_INSTANZEN,
    "seed_input": DEFAULT_SEED,
    "gitter_g": GITTER_STANDARD,
    "ziehung": ZIEHUNGEN[0],
    "regel": "erste",
    "mischung": 0.5,
}

# Presets: jede Zeile legt ALLE Regler fest (Schlüssel wie in DEFAULTS); `aktives_preset` vergleicht jeden Regler.
PRESETS = {
    "Von Hand: 3 × 3 Zuordnung": dict(DEFAULTS),
    "Von Hand: 3 × 3 Transport": dict(DEFAULTS, quelle=QUELLE_HAND_TRANSPORT),
    "Gleichstand: Gitter": dict(DEFAULTS, quelle=QUELLE_ZUFALL, familie="Gitter L1", n_slider=10, s_stufe=0.0, seed_input=7),
    "Übergang s = 0,001": dict(DEFAULTS, quelle=QUELLE_ZUFALL, familie="Punkte im Einheitsquadrat", n_slider=20, s_stufe=0.001, seed_input=11),
    "Viele Lager n = 20": dict(DEFAULTS, quelle=QUELLE_ZUFALL, familie="Punkte im Einheitsquadrat", n_slider=20, s_stufe=1.0, seed_input=3),
}
PRESET_HELP = {
    "Von Hand: 3 × 3 Zuordnung": "Drei Lager und drei Kunden mit je einer Einheit; die beste Zuordnung ist eindeutig und kostet 5 (die sechs möglichen Zuordnungen kosten 5, 6, 6, 7, 9 und 11).",
    "Von Hand: 3 × 3 Transport": "Angebote 9, 6, 5 und Nachfragen 8, 8, 4: der optimale Plan kostet 97, zwei der drei Lager sind gespalten.",
    "Gleichstand: Gitter": "Zehn Lager und zehn Kunden auf einem 6 × 6-Gitter, Kosten nach L1: in 98 von 100 Instanzen (Zufalls-Seed 7) gibt es mehrere optimale Pläne.",
    "Übergang s = 0,001": "Zwanzig Lager und Kunden, Massen mit dem Mischgewicht s = 0,001 (die einzelnen Massen weichen typisch um etwa 0,1 % vom Mittel ab): im Mittel 15,15 von 20 Lagern gespalten, aber nur 0,13 % der Masse (100 Instanzen, Zufalls-Seed 11).",
    "Viele Lager n = 20": "Zwanzig Lager und Kunden mit unabhängig gezogenen Massen (s = 1): im Mittel 9,81 von 20 Lagern gespalten, 29,71 % der Masse außerhalb des Hauptabnehmers (100 Instanzen, Zufalls-Seed 3).",
}
