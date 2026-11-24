"""Kennzahlen, Reihen, Verdikte und Kurztexte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Zufalls-Seeds."""

import numpy as np
import pytest

import ztr_constants as C
import ztr_evaluation as E
import ztr_scenario as SC
from ztr_simplex import loese_transport


# ----------------------------------------------------------------------------------------------------------------------------------
# Kennzahlen eines Plans
# ----------------------------------------------------------------------------------------------------------------------------------

PLAN2 = np.array([[8.0, 1.0, 0.0], [0.0, 6.0, 0.0], [0.0, 1.0, 4.0]])      # Handbeispiel 2


def test_kennzahlen_des_handbeispiels_2():
    assert E.zeilenzahlen(PLAN2).tolist() == [2, 1, 2]
    assert E.gespaltene_lager(PLAN2) == 2 and E.eintraege(PLAN2) == 5
    assert E.nebenmasse(PLAN2) == pytest.approx(0.1)              # (9 - 8) + (6 - 6) + (5 - 4) = 2 von 20
    assert not E.ist_monge(PLAN2) and not E.ist_permutation(PLAN2)


def test_kennzahlen_diagonale_permutation_und_nichtquadratischer_plan():
    diag = np.diag([8.0, 8.0, 4.0])
    assert E.ist_monge(diag) and E.gespaltene_lager(diag) == 0 and E.nebenmasse(diag) == 0.0
    assert E.ist_permutation(np.eye(3)[[1, 0, 2]] / 3)
    assert not E.ist_permutation(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))          # zwei Lager an denselben Kunden
    assert E.ist_monge(np.array([[2.0, 0.0], [1.0, 0.0]])) and not E.ist_permutation(np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))   # nicht quadratisch


def test_ein_lager_mit_menge_0_zaehlt_nicht_als_gespalten():
    plan = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert E.gespaltene_lager(plan) == 0 and E.ist_monge(plan) and E.nebenmasse(plan) == 0.0
    assert E.nebenmasse(np.zeros((2, 2))) == 0.0                   # Gesamtmasse 0: kein Teilen durch 0


def test_rauschgrenze_zaehlt_kleine_reste_nicht_mit():
    plan = np.array([[0.5, 1e-15], [0.0, 0.5]])
    assert E.gespaltene_lager(plan) == 0 and E.eintraege(plan) == 2


def test_gleiche_plaene_toleranz():
    assert E.gleiche_plaene(np.eye(3), np.eye(3) + 1e-12) and not E.gleiche_plaene(np.eye(3), np.eye(3)[[1, 0, 2]])


def test_mischung_liegt_im_polytop_und_t_0_und_1_geben_die_ausgangsplaene():
    p, q = np.eye(3), np.eye(3)[[1, 0, 2]]
    assert np.array_equal(E.mischung(p, q, 1.0), p) and np.array_equal(E.mischung(p, q, 0.0), q)
    m = E.mischung(p, q, 0.25)
    assert m.sum(axis=0).tolist() == [1, 1, 1] and (m >= 0).all() and m[0, 0] == 0.25


# ----------------------------------------------------------------------------------------------------------------------------------
# Ansicht und Reihen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_ansicht_handbeispiele(hand1, hand2):
    a = E.analysiere_instanz(hand2, "erste")
    assert (a.gespalten, a.eintraege, a.monge, a.permutation, a.eindeutig, a.fehler) == (2, 5, False, False, True, ())
    b = E.analysiere_instanz(hand1, "letzte")
    assert (b.gespalten, b.eintraege, b.monge, b.permutation, b.eindeutig) == (0, 3, True, True, True)


def test_analysiere_nimmt_handbeispiel_oder_zufallsinstanz():
    assert E.analysiere(E.Settings()).inst.name == C.QUELLE_HAND_ZUORDNUNG
    assert E.analysiere(E.Settings(quelle=C.QUELLE_HAND_TRANSPORT)).gespalten == 2
    z = E.analysiere(E.Settings(quelle=C.QUELLE_ZUFALL, familie="Punkte im Einheitsquadrat", n=12, s=1.0, seed=3))
    assert z.inst.n == 12 and z.fehler == () and z.eintraege == 23


def test_reihe_zuordnen_zaehlt_permutationen_und_vergleicht_mit_der_ungarischen_methode():
    r = E.reihe_zuordnen("ganzzahlig 0 bis 10", 6, 40, 5)
    assert r["instanzen"] == 40 and r["permutation"] == 40 and r["gleicher_wert"] == 40 and r["max_abweichung"] < 1e-9


def test_reihe_uebergang_s_0_ist_immer_eine_zuordnung_und_s_groesser_0_nie_ohne_spaltung():
    zeilen = E.reihe_uebergang("Punkte im Einheitsquadrat", 12, (0.0, 0.01, 1.0), 40, 2)
    assert [z["s"] for z in zeilen] == [0.0, 0.01, 1.0]
    assert zeilen[0]["monge"] == 40 and zeilen[0]["gespalten_mittel"] == 0.0 and zeilen[0]["voll"] == 0       # entartet: nur n Einträge
    for z in zeilen[1:]:
        assert z["monge"] == 0 and z["voll"] == 40 and z["gespalten_min"] >= 1                              # in dieser Reihe: n + m - 1 Einträge, immer gespalten
        assert z["gespalten_min"] <= z["gespalten_mittel"] <= z["gespalten_max"] <= 12
    assert zeilen[1]["neben_mittel"] < zeilen[2]["neben_mittel"]                                             # Nebenmasse wächst mit s


def test_reihe_uebergang_je_s_eigene_instanzen():
    zeilen = E.reihe_uebergang("stetig (Zufallskosten)", 8, (0.5, 0.5), 30, 4)
    assert zeilen[0]["neben_mittel"] != zeilen[1]["neben_mittel"]


def test_reihe_gleichstand_stetige_kosten_sind_eindeutig_und_ganzzahlige_nicht():
    stetig = E.reihe_gleichstand("stetig (Zufallskosten)", 8, 0.0, 30, 1)
    assert stetig["mehrere_optima"] == 0 and stetig["verschiedene_plaene"] == 0
    ganz = E.reihe_gleichstand("ganzzahlig 0 bis 2", 8, 0.0, 30, 1)
    assert ganz["mehrere_optima"] >= 25 and ganz["verschiedene_plaene"] >= 10 and ganz["max_wertabweichung"] < 1e-9
    assert ganz["gitter_seite"] == 0 and ganz["zweit_spanne_mittel"] > 0


def test_gleichstand_ansicht_gitter_l1_hat_mehrere_optima_und_zwei_verschiedene_plaene():
    ga = E.gleichstand_ansicht(E.Settings(quelle=C.QUELLE_ZUFALL, familie="Gitter L1", n=10, s=0.0, seed=7))
    assert not ga.eindeutig and ga.misch_a is not None and ga.misch_b is not None and ga.misch_a < ga.misch_b
    kosten = {round(lo.kosten, 9) for lo in ga.loesungen}
    assert len(kosten) == 1                                                                 # gleicher Wert
    assert not E.gleiche_plaene(ga.loesungen[ga.misch_a].plan, ga.loesungen[ga.misch_b].plan)


def test_gleichstand_ansicht_eindeutig_ohne_mischung():
    ga = E.gleichstand_ansicht(E.Settings(quelle=C.QUELLE_ZUFALL, familie="stetig (Zufallskosten)", n=6, s=0.0, seed=7))
    assert ga.eindeutig and ga.misch_a is None and ga.misch_b is None


# ----------------------------------------------------------------------------------------------------------------------------------
# Preise
# ----------------------------------------------------------------------------------------------------------------------------------

def test_preise_ansicht_handbeispiel_2(hand2):
    p = E.preise_ansicht(hand2)
    assert p["u"].tolist() == [0, -16, -15] and p["v"].tolist() == [4, 20, 19]
    assert p["primal"] == 97 and p["dual"] == 97
    assert p["schlupf"].tolist() == [[0, 0, 10], [32, 0, 26], [16, 0, 0]]
    assert p["preisformel"] == -11 and p["preisprobe"] == -11 and p["kosten_neu"] == 86


def test_dual_luecke_ist_klein():
    r = E.dual_luecke(30, 1, n=8)
    assert r["instanzen"] == 30 and r["luecke"] < 1e-12 and r["verletzung"] <= 1e-12


# ----------------------------------------------------------------------------------------------------------------------------------
# Verdikte und Kurztexte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_zahlformate():
    assert E.de(10.123, 1) == "10,1" and E.prozent(0.1) == "10,0 %" and E.prozent(0.00137, 2) == "0,14 %" and E.de_e(7.1e-15) == "7,1e-15"
    assert [E.s_text(s) for s in C.S_STUFEN] == ["0", "0,001", "0,01", "0,1", "1"]


R1 = {"familie": "Gitter L1", "n": 10, "instanzen": 100, "seed": 35, "permutation": 100, "gleicher_wert": 100, "max_abweichung": 7.1e-15}


def test_verdikt_zuordnen_alle_permutationen_ist_success_und_nennt_die_grenze():
    kind, text = E.verdict_zuordnen(R1)
    assert kind == "success"
    assert "in **100 von 100** Instanzen ist die Ecke eine Permutationsmatrix" in text and "Gitter L1, n = 10, 100 Instanzen" in text
    assert "Ungarischen Methode in 100 von 100" in text and "7,1e-15" in text and "ohne Crossover ist nicht gemessen" in text


def test_verdikt_zuordnen_nicht_alle_permutationen_ist_warning():
    kind, text = E.verdict_zuordnen(dict(R1, permutation=97, gleicher_wert=99))
    assert kind == "warning" and "nur in **97 von 100**" in text and "Ungarischen Methode in 99 von 100" in text


def _zeile(s, gesp, neben, monge, voll=100, n=20, k=100, lo=13, hi=18):
    return {"s": s, "familie": "Punkte im Einheitsquadrat", "n": n, "instanzen": k, "seed": 1, "gespalten_mittel": gesp, "gespalten_min": lo, "gespalten_max": hi,
            "neben_mittel": neben, "monge": monge, "voll": voll}


ZEILEN = [_zeile(0.0, 0.0, 0.0, 100, 0, lo=0, hi=0), _zeile(0.001, 15.2, 0.0014, 0), _zeile(1.0, 9.9, 0.298, 0, lo=6, hi=14)]


def test_verdikt_uebergang_nennt_zaehlung_und_masse_nebeneinander_und_den_befund_ohne_ausnahme():
    kind, text = E.verdict_uebergang(ZEILEN)
    assert kind == "info"
    assert "Punkte im Einheitsquadrat, n = m = 20, 100 Instanzen je s" in text
    assert "s = 0: 0,0 von 20 Lagern gespalten, 0,00 % der Masse" in text
    assert "s = 0,001: 15,2 von 20 Lagern gespalten, 0,14 % der Masse außerhalb des Hauptabnehmers" in text
    assert "s = 1: 9,9 von 20 Lagern gespalten, 29,8 % der Masse" in text
    assert "In keiner der 100 Instanzen je s > 0 liefert jedes Lager an genau einen Kunden." in text


def test_verdikt_uebergang_mit_ausnahme_nennt_die_zaehlung_statt_keiner():
    zeilen = [ZEILEN[0], _zeile(0.001, 15.2, 0.0014, 3), _zeile(1.0, 9.9, 0.298, 0)]
    text = E.verdict_uebergang(zeilen)[1]
    assert "In 3 von 200 Instanzen mit s > 0 liefert jedes Lager an genau einen Kunden." in text and "In keiner" not in text


def test_verdikt_uebergang_nur_s_0_hat_keinen_schlusssatz():
    text = E.verdict_uebergang([ZEILEN[0]])[1]
    assert text.endswith(".") and "s > 0" not in text


def test_kurztexte_nennen_die_kernzahlen():
    assert "**100 von 100** Instanzen eine Permutationsmatrix (Gitter L1, n = 10)" in E.kurz_zuordnen(R1)
    t = E.kurz_uebergang(ZEILEN)
    assert t.startswith("s = 0: 0,0 von 20 Lagern gespalten, 0,00 % der Masse") and "s = 0,001: 15,2 von 20" in t and "s = 1: 9,9 von 20 Lagern gespalten, 29,8 %" in t
    assert t.count("s = ") == 3 and "Mittel über 100 Instanzen je s" in t
    nur = E.kurz_uebergang([ZEILEN[0]])
    assert nur.count("s = ") == 1
    r3 = {"familie": "Gitter L1", "n": 10, "s": 0.0, "instanzen": 100, "mehrere_optima": 98, "verschiedene_plaene": 92}
    assert E.kurz_gleichstand(r3).startswith("In **98 von 100** Instanzen gibt es mehrere optimale Pläne (Gitter L1, n = 10)") and "in 92 von 100 Instanzen verschiedene Pläne" in E.kurz_gleichstand(r3)


R3 = {"familie": "Gitter L1", "n": 10, "s": 0.0, "instanzen": 100, "seed": 7, "gitter_seite": 6, "ziehung": "ohne Zurücklegen", "mehrere_optima": 98, "verschiedene_plaene": 92,
      "max_wertabweichung": 0.0, "zweit_spanne_mittel": 0.13}


def test_verdikt_gleichstand_nennt_familie_gitter_ziehung_n_und_zahl_der_instanzen():
    kind, text = E.verdict_gleichstand(R3)
    assert kind == "info"
    assert text.startswith("Zuordnung, Gitter L1, n = 10, 100 Instanzen:") and "in **98 von 100** Instanzen gibt es mehrere optimale Pläne" in text
    assert "in 92 von 100 Instanzen verschiedene Pläne" in text and "Gitterseite 6, ohne Zurücklegen." in text and "keinen Fehler, nur eine Auswahl" in text


def test_verdikt_gleichstand_transport_und_stetige_kosten_ohne_gleichstand():
    kind, text = E.verdict_gleichstand(dict(R3, s=0.1, familie="stetig (Zufallskosten)", gitter_seite=0, mehrere_optima=0, verschiedene_plaene=0))
    assert kind == "success" and text.startswith("Transport (s = 0,1), stetig (Zufallskosten), n = 10") and "Gitterseite" not in text and "**0 von 100**" in text


def test_verdikt_ansicht_zweig_monge_und_zweig_gespalten(hand1, hand2):
    kind, text = E.verdict_ansicht(E.analysiere_instanz(hand1, "erste"))
    assert kind == "success" and "Monge-Abbildung" in text and "**0 von 3** Lagern" in text and "3 Einträge im Plan (3 + 3 − 1 = 5" in text
    kind, text = E.verdict_ansicht(E.analysiere_instanz(hand2, "erste"))
    assert kind == "warning" and "**2 von 3** Lagern sind gespalten" in text and "**10,0 %** der Masse" in text and "5 Einträge im Plan (3 + 3 − 1 = 5" in text


def test_zufalls_zweig_der_verdikte_mit_echter_reihe():
    zeilen = E.reihe_uebergang("stetig (Zufallskosten)", 6, (0.0, 1.0), 10, 3)
    kind, text = E.verdict_uebergang(zeilen)
    assert "In keiner der 10 Instanzen je s > 0" in text and "s = 0: 0,0 von 6" in text


def test_verdikt_gleichstand_schlusssatz_haengt_von_der_messung_ab():
    """Der Schlusssatz „keinen Fehler, nur eine Auswahl“ steht nur, wenn die Regeln verschiedene Pläne wählen (Befund der Aussagenprüfung)."""
    eindeutig = dict(R3, mehrere_optima=0, verschiedene_plaene=0, familie="Punkte im Einheitsquadrat", gitter_seite=0, s=0.001, n=20)
    kind, text = E.verdict_gleichstand(eindeutig)
    assert kind == "success" and "keinen Fehler" not in text and "hängt hier von der Regel ab" not in text
    assert text.endswith("Der Plan ist in allen 100 Instanzen eindeutig; die Wahl der Regel ändert nichts.")
    kind, text = E.verdict_gleichstand(R3)
    assert kind == "info" and text.endswith("Es gibt keinen Fehler, nur eine Auswahl: welche Ecke herauskommt, hängt hier von der Regel ab.")
    kind, text = E.verdict_gleichstand(dict(R3, mehrere_optima=5, verschiedene_plaene=0))
    assert kind == "info" and "keinen Fehler" not in text and "wählen hier dieselbe Ecke" in text and "hängt hier von der Regel ab" not in text


def test_zweit_text_in_der_ansicht_ohne_verschiedene_plaene():
    assert E.zweit_text(dict(R3, verschiedene_plaene=0, zweit_spanne_mittel=0.0)) == "–" and E.zweit_text(R3) == "13,0 %"
