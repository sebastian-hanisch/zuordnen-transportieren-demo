"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a).

Die Messreihen liegen in `tools/messreihen.json` (erzeugt von `tools/messreihen.py` mit dem Code der Demo). Dieser Test macht zwei Dinge:
(1) die README-Zahlen werden aus der Datei mit denselben Formatfunktionen gebildet und müssen wörtlich in der README stehen (so kann keine Zahl der README abweichen);
(2) die kleinen Reihen werden neu gerechnet und als Band mit der Datei verglichen (Zählungen ±1, Mittel ±0,05), weil die Wahl der Ecke bei Gleichstand auf anderen
Plattformen anders ausfallen kann. Die großen Reihen (1 000 Instanzen, n = 40) stehen nur in der Datei."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ztr_constants as C
import ztr_evaluation as E

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
J = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))
TAB = "Punkte im Einheitsquadrat"


def ganz(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def zeile_uebergang(r: dict) -> str:
    neben = E.prozent(r["neben_mittel"], 3 if r["neben_mittel"] < 0.01 else 2)
    return f"| {E.s_text(r['s'])} | {E.de(r['gespalten_mittel'], 2)} [{r['gespalten_min']}, {r['gespalten_max']}] | {neben} | {r['monge']} von {r['instanzen']} | {r['voll']} von {r['instanzen']} |"


def zeile_gleichstand(r: dict) -> str:
    ort = r["gitter_seite"] or "–"
    ziehung = r["ziehung"] if r["gitter_seite"] else "–"
    return (f"| {r['familie']} | {r['n']} | {E.s_text(r['s'])} | {ort} | {ziehung} | {r['instanzen']} | {r['mehrere_optima']} von {r['instanzen']} | "
            f"{r['verschiedene_plaene']} von {r['instanzen']} | {E.zweit_text(r)} |")


def _gleich(a, b, band):
    return abs(a - b) <= band


# ----------------------------------------------------------------------------------------------------------------------------------
# 1. Zuordnung
# ----------------------------------------------------------------------------------------------------------------------------------

def test_meta_der_messreihen_ist_der_stand_der_demo():
    assert J["meta"] == {"seed": 2026, "regel_reihen": C.REGEL_REIHEN} and "Zufalls-Seed 2026" in README


def test_bland_ruckfall_nach_25_entarteten_schritten():
    import ztr_simplex as S
    assert S.ZAEHLER_BLAND == 25 and "nach 25 entarteten Schritten in Folge" in README


def test_zuordnen_gesamtzahl_und_wertabweichung():
    zeilen = J["zuordnen"]
    assert len(zeilen) == 28 and {r["familie"] for r in zeilen} == set(C.FAMILIEN)                                     # sieben Familien, vier Größen
    assert sorted({(r["n"], r["instanzen"]) for r in zeilen}) == [(5, 200), (10, 200), (20, 100), (40, 20)]
    gesamt = sum(r["instanzen"] for r in zeilen)
    assert gesamt == 3640 and all(r["permutation"] == r["instanzen"] == r["gleicher_wert"] for r in zeilen)
    assert f"**{ganz(gesamt)} von {ganz(gesamt)} Instanzen**" in README and f"in {ganz(gesamt)} von {ganz(gesamt)} Instanzen (größte Abweichung" in README
    abw = max(r["max_abweichung"] for r in zeilen)
    assert E.de_e(abw) == "5,7e-14" and f"größte Abweichung {E.de_e(abw)})" in README
    assert "200 / 200 / 100 / 20 Instanzen" in README


def test_zuordnen_n5_neu_gerechnet():
    for fam in C.FAMILIEN:
        alt = next(r for r in J["zuordnen"] if r["familie"] == fam and r["n"] == 5)
        neu = E.reihe_zuordnen(fam, 5, 200, J["meta"]["seed"])
        assert neu["permutation"] == neu["gleicher_wert"] == 200 and _gleich(neu["permutation"], alt["permutation"], 1)
        assert neu["max_abweichung"] < 1e-9


# ----------------------------------------------------------------------------------------------------------------------------------
# 2. Übergang
# ----------------------------------------------------------------------------------------------------------------------------------

def test_uebergang_tabelle_der_readme():
    zeilen = J["uebergang"]
    assert [r["s"] for r in zeilen] == [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0] and all(r["familie"] == TAB and r["n"] == 20 and r["instanzen"] == 100 for r in zeilen)
    for r in zeilen:
        assert zeile_uebergang(r) in README, r["s"]
    nur_s0 = zeilen[0]
    assert (nur_s0["gespalten_mittel"], nur_s0["monge"], nur_s0["voll"]) == (0.0, 100, 0)                                # s = 0: entartete Zuordnung
    assert all(r["monge"] == 0 and r["voll"] == 100 and r["gespalten_min"] >= 1 for r in zeilen[1:])                      # s > 0: nie ohne Spaltung, immer n + m - 1 Einträge
    s0001 = zeilen[1]
    assert 0.7 <= s0001["gespalten_mittel"] / 20 <= 0.8 and "etwa drei Viertel der Lager gespalten" in README             # „etwa drei Viertel“: 15,2 von 20 = 76 %
    assert 0.0013 <= s0001["neben_mittel"] <= 0.0015 and "etwa 0,14 %" in README
    assert [E.s_text(r["s"]) for r in zeilen] == ["0", "0,001", "0,01", "0,03", "0,1", "0,3", "1"]


def test_uebergang_neu_gerechnet():
    neu = E.reihe_uebergang(TAB, 20, tuple(r["s"] for r in J["uebergang"]), 100, J["meta"]["seed"])
    for a, b in zip(neu, J["uebergang"]):
        assert _gleich(a["gespalten_mittel"], b["gespalten_mittel"], 0.05) and _gleich(a["neben_mittel"], b["neben_mittel"], 0.002), a["s"]
        assert _gleich(a["monge"], b["monge"], 1) and _gleich(a["voll"], b["voll"], 1)


def test_nebenmasse_durch_s():
    zeilen = {r["s"]: r for r in J["uebergang"]}
    verhaeltnisse = [zeilen[s]["neben_mittel"] / s for s in (0.001, 0.01, 0.03, 0.1)]
    text = " / ".join(E.de(v, 2) for v in verhaeltnisse)
    assert text == "1,42 / 1,32 / 1,45 / 1,29" and f"ist {text} für" in README
    assert zeilen[0.3]["neben_mittel"] / 0.3 < 0.8 and zeilen[1.0]["neben_mittel"] < 0.3                                  # danach nicht mehr proportional zu s
    assert all(1.28 <= v <= 1.46 for v in verhaeltnisse) and "also etwa 1,3 bis 1,45" in README
    a, b = J["uebergang_1000"]                                                                                           # mit 1 000 Instanzen: 1,39 und 1,37
    assert (E.de(a["neben_mittel"] / a["s"], 2), E.de(b["neben_mittel"] / b["s"], 2)) == ("1,39", "1,37") and "mit 1 000 Instanzen ergeben sich unten 1,39 und 1,37" in README


def test_uebergang_zusatzreihen():
    r1000 = J["uebergang_1000"]
    assert [(r["s"], r["instanzen"], r["n"]) for r in r1000] == [(0.001, 1000, 20), (0.03, 1000, 20)] and sum(r["monge"] for r in r1000) == 0
    a, b = r1000
    assert f"$s = 0{{,}}001$ gibt {E.de(a['gespalten_mittel'], 2)} gespaltene Lager und {E.prozent(a['neben_mittel'], 3)} Nebenmasse" in README
    assert f"$s = 0{{,}}03$ gibt {E.de(b['gespalten_mittel'], 2)} und {E.prozent(b['neben_mittel'], 3)}" in README and "in keiner dieser 2 000 Instanzen" in README
    st = J["uebergang_stetig"][0]
    assert (st["familie"], st["s"], st["instanzen"], st["monge"]) == (C.FAMILIEN[1], 1.0, 200, 0)
    assert f"{E.de(st['gespalten_mittel'], 2)} gespaltene Lager [{st['gespalten_min']}, {st['gespalten_max']}], {E.prozent(st['neben_mittel'], 3)} Nebenmasse, Monge-Abbildung in 0 von 200" in README
    ns = J["uebergang_n"]
    assert [(r["n"], r["instanzen"], r["s"]) for r in ns] == [(5, 60, 1.0), (10, 60, 1.0), (20, 60, 1.0), (40, 60, 1.0)]
    assert " / ".join(E.de(r["gespalten_mittel"], 2) for r in ns) == "2,48 / 4,93 / 9,63 / 20,18" and "2,48 / 4,93 / 9,63 / 20,18 gespaltene Lager" in README
    assert " / ".join(E.de(100 * r["neben_mittel"], 1) for r in ns) == "26,6 / 29,0 / 30,5 / 29,7" and "26,6 / 29,0 / 30,5 / 29,7 % Nebenmasse" in README
    assert all(0.45 <= r["gespalten_mittel"] / r["n"] <= 0.55 for r in ns[1:])                                           # „etwa die Hälfte“ (n = 10, 20, 40)
    assert all(r["monge"] == 0 for r in ns)


# ----------------------------------------------------------------------------------------------------------------------------------
# 3. Gleichstände
# ----------------------------------------------------------------------------------------------------------------------------------

def test_gleichstand_tabelle_der_readme():
    alle = J["gleichstand"] + J["gleichstand_gitter"] + J["gleichstand_transport"]
    assert len(alle) == 5 + 6 + 2
    for r in alle:
        assert zeile_gleichstand(r) in README, (r["familie"], r["gitter_seite"], r["ziehung"])
    zeilen = {(r["familie"], r["gitter_seite"], r["ziehung"], r["s"]): r for r in alle}
    assert zeilen[(C.FAMILIEN[1], 0, "ohne Zurücklegen", 0.0)]["mehrere_optima"] == 0 == zeilen[(C.FAMILIEN[1], 0, "ohne Zurücklegen", 1.0)]["mehrere_optima"]     # stetig: eindeutig
    assert all(r["max_wertabweichung"] < 1e-13 for r in alle) and "größte Abweichung unter $10^{-13}$" in README
    assert zeilen[("Gitter L2²", 20, "mit Zurücklegen", 0.0)]["mehrere_optima"] == 53 and zeilen[("Gitter L2²", 6, "mit Zurücklegen", 0.0)]["mehrere_optima"] == 150
    assert "(Gitter L2² mit Zurücklegen: 150 von 150 bei Seite 6, 53 von 150 bei Seite 20)" in README


def test_gleichstand_neu_gerechnet():
    for alt in J["gleichstand"][:3] + [J["gleichstand"][3]]:
        neu = E.reihe_gleichstand(alt["familie"], alt["n"], alt["s"], alt["instanzen"], J["meta"]["seed"])
        assert _gleich(neu["mehrere_optima"], alt["mehrere_optima"], 1) and _gleich(neu["verschiedene_plaene"], alt["verschiedene_plaene"], 2), alt["familie"]
        assert neu["max_wertabweichung"] < 1e-9 and _gleich(neu["zweit_spanne_mittel"], alt["zweit_spanne_mittel"], 0.02)


def test_gleichstand_aussagen_der_readme_gelten_in_allen_zeilen():
    alle = J["gleichstand"] + J["gleichstand_gitter"] + J["gleichstand_transport"]
    assert all(r["verschiedene_plaene"] <= r["mehrere_optima"] for r in alle)       # verschiedene Pläne gibt es nur bei Gleichstand
    stetig = [r for r in alle if r["familie"] == C.FAMILIEN[1]]
    assert len(stetig) == 2 and all(r["mehrere_optima"] == r["verschiedene_plaene"] == 0 for r in stetig)
    assert sum(1 for r in alle if r["zweit_spanne_mittel"] > 0) == len(alle) - 2    # in allen Zeilen außer den beiden stetigen streut die zweite Kennzahl


FREMD = json.loads((ROOT / "tools" / "messreihen_fremdloeser.json").read_text(encoding="utf-8"))


def test_fremdloeser_reihen_der_readme():
    reihen = FREMD["reihen"]
    assert [r["familie"] for r in reihen] == [C.FAMILIEN[1], C.FAMILIEN[2], C.FAMILIEN[3], C.FAMILIEN[4], "Gitter L1", "Gitter L2²"]
    assert all(r["n"] == 10 and r["instanzen"] == 150 and r["gleicher_wert"] == 150 for r in reihen) and sum(r["instanzen"] for r in reihen) == 900
    assert " / ".join(str(r["verschiedene_plaene"]) for r in reihen) == "0 / 14 / 86 / 148 / 148 / 113"
    assert "mindestens zwei der fünf Pläne sind verschieden in 0 / 14 / 86 / 148 / 148 / 113 von 150 Instanzen" in README and "In jeder dieser 900 Instanzen" in README
    m = FREMD["meta"]
    assert f"numpy {m['numpy']}, scipy {m['scipy']}, networkx {m['networkx']}, OR-Tools {m['ortools'].rsplit('.', 1)[0]}" in README and len(m["loeser"]) == 5


def test_fremdloeser_neu_gerechnet():
    """Band: der Wert ist bei allen fünf Lösern gleich (exakt), der Anteil verschiedener Pläne liegt in der Nähe der Datei; die Wahl der Ecke kann sich je Version ändern."""
    pytest.importorskip("networkx")
    pytest.importorskip("ortools")
    pytest.importorskip("scipy")
    sys.path.insert(0, str(ROOT / "tools"))
    import messreihen_fremdloeser as F
    for alt in FREMD["reihen"]:
        if alt["familie"] not in (C.FAMILIEN[1], C.FAMILIEN[3]):
            continue
        neu = F.reihe(alt["familie"], 10, 30)
        assert neu["gleicher_wert"] == 30
        anteil_alt, anteil_neu = alt["verschiedene_plaene"] / alt["instanzen"], neu["verschiedene_plaene"] / 30
        assert abs(anteil_alt - anteil_neu) <= 0.3, (alt["familie"], anteil_alt, anteil_neu)


# ----------------------------------------------------------------------------------------------------------------------------------
# 4. Dualität und Handwerte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_dualitaet_zeile():
    d = J["dualitaet"]
    assert d["instanzen"] == 100 and d["groessen"] == [3, 8, 20, 40]
    assert E.de_e(d["luecke"]) == "4,4e-16" and E.de_e(d["verletzung"]) == "5,6e-17"
    assert f"größte Abweichung |Primalwert − Dualwert| {E.de_e(d['luecke'])} und größte Verletzung von $u_i + v_j \\le c_{{ij}}$ {E.de_e(d['verletzung'])}" in README
    neu = E.dual_luecke(10, 5, n=10)
    assert neu["luecke"] < 1e-12 and neu["verletzung"] < 1e-12


def test_handwerte_der_readme(hand1, hand2):
    p = E.preise_ansicht(hand2)
    assert (p["primal"], p["dual"], p["preisformel"], p["preisprobe"], p["kosten_neu"]) == (97, 97, -11, -11, 86)
    assert p["u"].tolist() == [0, -16, -15] and p["v"].tolist() == [4, 20, 19]
    for text in ("Optimum 97", "$u = (0, -16, -15)$ und $v = (4, 20, 19)$", "$u_3 + v_1 = -11$ (97 → 86", "Dualwert 97", "aller 16 Ecken der geänderten Aufgabe", "Rang 5 bei sechs Unbekannten"):
        assert text in README, text
    a = E.analysiere_instanz(hand2, "erste")
    assert (a.gespalten, a.eintraege, round(20 * a.neben)) == (2, 5, 2) and "Nebenmasse 2 von 20 = 10 %, 5 Einträge" in README
    b = E.analysiere_instanz(hand1, "erste")
    assert b.loesung.kosten == 5 and "die beste ist eindeutig und kostet 5" in README and "5 / 6 / 6 / 7 / 9 / 11" in README
    assert "Kosten 80, drei Einträge" in README and "(Kosten 0)" in README and "Kosten 3" in README
    assert "0, 2 und 1 für die beiden Ecken und die Mitte" in README


def test_preset_hilfetexte_nennen_gemessene_zahlen():
    pr = J["presets"]
    gleich = pr["Gleichstand: Gitter"]["gleichstand"]
    assert f"in {gleich['mehrere_optima']} von {gleich['instanzen']} Instanzen (Zufalls-Seed {gleich['seed']}) gibt es mehrere optimale Pläne" in C.PRESET_HELP["Gleichstand: Gitter"]
    assert gleich["gitter_seite"] == 6 and "6 × 6-Gitter" in C.PRESET_HELP["Gleichstand: Gitter"]
    for name, s in (("Übergang s = 0,001", 0.001), ("Viele Lager n = 20", 1.0)):
        zeile = next(r for r in pr[name]["uebergang"] if r["s"] == s)
        hilfe = C.PRESET_HELP[name]
        assert f"im Mittel {E.de(zeile['gespalten_mittel'], 2)} von {zeile['n']} Lagern gespalten" in hilfe, name
        assert E.prozent(zeile["neben_mittel"], 2) in hilfe and f"{zeile['instanzen']} Instanzen, Zufalls-Seed {zeile['seed']}" in hilfe, name
    a = E.analysiere(E.settings_aus_zustand(C.PRESETS["Von Hand: 3 × 3 Zuordnung"]))
    b = E.analysiere(E.settings_aus_zustand(C.PRESETS["Von Hand: 3 × 3 Transport"]))
    assert a.loesung.kosten == 5 and "kostet 5" in C.PRESET_HELP["Von Hand: 3 × 3 Zuordnung"] and a.eindeutig
    assert b.loesung.kosten == 97 and b.gespalten == 2 and "kostet 97, zwei der drei Lager sind gespalten" in C.PRESET_HELP["Von Hand: 3 × 3 Transport"]


# ----------------------------------------------------------------------------------------------------------------------------------
# Testzahl und Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_testzahl_der_readme():
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    zahl = int(re.search(r"(\d+) tests? collected", r.stdout).group(1))
    assert f"**Testzahl:** {zahl} Tests (`pytest --collect-only -q tests`)" in README, zahl


def test_readme_und_app_nennen_keine_planbegriffe_und_keine_umgebungsaussagen():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (README, app):
        for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Saatwert", "trägt", "tragen", "Vormessung", "Mutationstest"):
            assert verboten not in text, verboten
    assert "Orakel" not in app                                                                                           # die Orakel-Wörter gehören in Tests und README, nicht in App-Texte
    assert "Zufalls-Seed" in app and "Zufallsseed" not in README


# ----------------------------------------------------------------------------------------------------------------------------------
# Befunde der Aussagenprüfung (2026-11-24): Streuung der Quotienten, Massenabweichung, Dualbereich, Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_streuung_des_quotienten_nebenmasse_durch_s_ist_stichprobenrauschen():
    """Andere Zufalls-Seeds (je 100 Instanzen) streuen den Quotienten um etwa 0,05 (Band 0,02 bis 0,09); "±0,05" der README."""
    import numpy as np
    q = []
    for seed in range(100, 108):
        z = E.reihe_uebergang(TAB, 20, (0.001,), 100, seed)[0]
        q.append(z["neben_mittel"] / z["s"])
    assert 0.02 <= float(np.std(q, ddof=1)) <= 0.09 and 1.2 <= float(np.mean(q)) <= 1.55
    assert "um etwa ±0,05" in README


def test_einzelmassen_weichen_bei_s_0_001_typisch_um_ein_zehntel_prozent_ab():
    import numpy as np
    from ztr_scenario import make_instanz
    rel = []
    for i in range(100):
        inst = make_instanz(TAB, 20, 20, 0.001, 11, C.ZWECK_UEBERGANG, i)
        rel.extend((inst.a * 20 - 1).tolist() + (inst.b * 20 - 1).tolist())
    assert 0.0008 <= float(np.std(rel)) <= 0.0012                                  # typische Abweichung etwa 0,1 % (Standardabweichung)
    assert "typisch um etwa 0,1 % vom Mittel ab" in README and "typisch um etwa 0,1 % vom Mittel" in C.PRESET_HELP["Übergang s = 0,001"]


def test_preise_sind_bei_entartung_nicht_bis_auf_eine_konstante_bestimmt(hand1):
    """Handbeispiel Zuordnung 3 x 3 (entartet): bei u_1 = 0 sind u_2 in [-2, -1] und u_3 in [-1, 0] optimal (Dual-LP); im Handbeispiel Transport ist es eindeutig."""
    linprog = pytest.importorskip("scipy.optimize").linprog
    import numpy as np
    from ztr_scenario import handbeispiel_transport

    def bereich(inst, k, vorzeichen):
        n, m = inst.kosten.shape
        zeilen = []
        for i in range(n):
            for j in range(m):
                z = np.zeros(n + m)
                z[i], z[n + j] = 1.0, 1.0
                zeilen.append(z)
        gebunden = [(0, 0)] + [(None, None)] * (n + m - 1)
        rechts = np.concatenate([inst.a, inst.b])
        wert = linprog(-rechts, A_ub=np.array(zeilen), b_ub=inst.kosten.ravel(), bounds=gebunden).fun
        ziel = np.zeros(n + m)
        ziel[k] = vorzeichen
        r = linprog(ziel, A_ub=np.array(zeilen), b_ub=inst.kosten.ravel(), A_eq=[rechts], b_eq=[-wert], bounds=gebunden)
        return vorzeichen * r.fun

    assert (round(bereich(hand1, 1, 1), 6), round(bereich(hand1, 1, -1), 6)) == (-2.0, -1.0)
    assert (round(bereich(hand1, 2, 1), 6), round(bereich(hand1, 2, -1), 6)) == (-1.0, 0.0)
    h2 = handbeispiel_transport()
    assert (round(bereich(h2, 1, 1), 6), round(bereich(h2, 1, -1), 6)) == (-16.0, -16.0)
    assert (round(bereich(h2, 2, 1), 6), round(bereich(h2, 2, -1), 6)) == (-15.0, -15.0)
    assert "bei Entartung, etwa bei gleichen Massen, gibt es weitere optimale Preise" in README


def test_zweit_text_zeigt_strich_ohne_verschiedene_plaene():
    assert E.zweit_text({"verschiedene_plaene": 0, "zweit_spanne_mittel": 0.0}) == "–"
    assert E.zweit_text({"verschiedene_plaene": 3, "zweit_spanne_mittel": 0.132}) == "13,2 %"
    stetig = [r for r in J["gleichstand"] + J["gleichstand_transport"] if r["familie"] == C.FAMILIEN[1]]
    assert len(stetig) == 2 and all(E.zweit_text(r) == "–" for r in stetig)
    assert README.count("| 0 von 300 | 0 von 300 | – |") == 1 and README.count("| 0 von 150 | 0 von 150 | – |") == 1


def test_behauptungen_der_app_und_der_hilfen_sind_abgeschwaecht():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for alt in ("bei verschiedenen Mengen spaltet", "Bei verschiedenen Mengen **spaltet**", "entscheidet sie, welche optimale Ecke", "nur bis auf eine Konstante bestimmt |", "wählt bei Gleichstand die Mitte der Fläche",
                "sucht die nächste Matrix", "exakter Eindeutigkeitstest je Instanz", "nur der Plan nicht"):
        assert alt not in app, alt
    hilfen = C.FAMILIEN_HILFE["ganzzahlig 0 bis 2"] + " ".join(C.PRESET_HELP.values())
    for alt in ("fast immer Gleichstände", "Die Rechnung steht im Expander", "Massen nur um ein Tausendstel verschieden"):
        assert alt not in hilfen, alt
    assert "Bei verschiedenen Mengen **kann** der Plan Lager" in app and "bei Entartung (etwa bei gleichen Massen) gibt es weitere optimale Preise" in app
    assert "viele Gleichstände (ab n = 10 fast immer)" in C.FAMILIEN_HILFE["ganzzahlig 0 bis 2"]
    assert "(die sechs möglichen Zuordnungen kosten 5, 6, 6, 7, 9 und 11)" in C.PRESET_HELP["Von Hand: 3 × 3 Zuordnung"]


def test_familien_hilfe_ganzzahlig_0_bis_2_gleichstaende_nach_n():
    """FAMILIEN_HILFE: bei kleinem n nicht fast immer (n = 3: 37 von 100, n = 5: 72 von 100), ab n = 10 fast immer; Zählungen als Band."""
    r3 = E.reihe_gleichstand(C.FAMILIEN[4], 3, 0.0, 100, J["meta"]["seed"])
    r5 = E.reihe_gleichstand(C.FAMILIEN[4], 5, 0.0, 100, J["meta"]["seed"])
    r10 = E.reihe_gleichstand(C.FAMILIEN[4], 10, 0.0, 30, J["meta"]["seed"])
    assert 20 <= r3["mehrere_optima"] <= 55 and 55 <= r5["mehrere_optima"] <= 90 and r10["mehrere_optima"] >= 28
