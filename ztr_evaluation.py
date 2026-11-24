"""Kennzahlen, Reihen über feste Instanzen und Verdikte (reine Funktionen). Zählung und Masse stehen immer nebeneinander.

Zählweise: „gespaltenes Lager“ = Zeile des Plans mit mehr als einem Eintrag über TOL_MENGE (Gesamtmasse 1 vorausgesetzt, bei Handbeispielen auf die Gesamtmenge
bezogen); „Nebenmasse“ = Anteil der Gesamtmasse, den ein Lager nicht an seinen größten Abnehmer liefert; „Monge-Abbildung“ = kein Lager gespalten.
Zähleinheit aller Reihen ist die Instanz; jede Reihe nennt Familie, n, Zahl der Instanzen und Zufalls-Seed."""

from dataclasses import dataclass

import numpy as np

import ztr_constants as C
from ztr_eindeutigkeit import ist_eindeutig
from ztr_scenario import Instanz, instanz_aus_einstellungen, instanz_rng, make_instanz
from ztr_simplex import TOL_MENGE, Loesung, loese_transport, pruefe_loesung
from ztr_zuordnung import ungarisch

GLEICH = 1e-9     # Pläne gelten als gleich, wenn sich kein Eintrag um mehr als GLEICH · Gesamtmenge unterscheidet


@dataclass(frozen=True)
class Settings:
    quelle: str = C.DEFAULTS["quelle"]
    familie: str = C.DEFAULTS["familie"]
    n: int = C.DEFAULTS["n_slider"]
    s: float = C.DEFAULTS["s_stufe"]
    instanzen: int = C.DEFAULTS["instanzen"]
    seed: int = C.DEFAULTS["seed_input"]
    gitter: int = C.DEFAULTS["gitter_g"]
    ziehung: str = C.DEFAULTS["ziehung"]
    regel: str = C.DEFAULTS["regel"]
    mischung: float = C.DEFAULTS["mischung"]


def settings_aus_zustand(zustand) -> Settings:
    """Settings aus dem Session State (Schlüssel wie in `ztr_constants.DEFAULTS`)."""
    d = {k: zustand.get(k, v) for k, v in C.DEFAULTS.items()}
    return Settings(d["quelle"], d["familie"], int(d["n_slider"]), float(d["s_stufe"]), int(d["instanzen"]), int(d["seed_input"]), int(d["gitter_g"]),
                    d["ziehung"], d["regel"], float(d["mischung"]))


# ----------------------------------------------------------------------------------------------------------------------------------
# Kennzahlen eines Plans
# ----------------------------------------------------------------------------------------------------------------------------------

def zeilenzahlen(plan: np.ndarray) -> np.ndarray:
    """Zahl der Einträge über der Rauschgrenze je Lager (Zeile)."""
    schwelle = TOL_MENGE * max(1.0, float(plan.sum()))
    return (plan > schwelle).sum(axis=1)


def gespaltene_lager(plan: np.ndarray) -> int:
    return int((zeilenzahlen(plan) > 1).sum())


def nebenmasse(plan: np.ndarray) -> float:
    """Anteil der Gesamtmasse, den ein Lager nicht an seinen größten Abnehmer liefert (0 bis 1)."""
    gesamt = float(plan.sum())
    return 0.0 if gesamt <= 0 else float((plan.sum(axis=1) - plan.max(axis=1)).sum() / gesamt)


def eintraege(plan: np.ndarray) -> int:
    return int((plan > TOL_MENGE * max(1.0, float(plan.sum()))).sum())


def ist_monge(plan: np.ndarray) -> bool:
    """Kein Lager mit Menge größer null liefert an mehr als einen Kunden."""
    return bool((zeilenzahlen(plan) <= 1).all())


def ist_permutation(plan: np.ndarray) -> bool:
    """Genau ein Eintrag je Zeile und je Spalte (gleiche Massen vorausgesetzt: eine Permutationsmatrix mal Masse)."""
    schwelle = TOL_MENGE * max(1.0, float(plan.sum()))
    fest = plan > schwelle
    return bool(plan.shape[0] == plan.shape[1] and (fest.sum(axis=0) == 1).all() and (fest.sum(axis=1) == 1).all())


def gleiche_plaene(p, q) -> bool:
    return bool(np.abs(p - q).max(initial=0.0) <= GLEICH * max(1.0, float(p.sum())))


# ----------------------------------------------------------------------------------------------------------------------------------
# Ansicht: eine Instanz
# ----------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Ansicht:
    inst: Instanz
    loesung: Loesung
    eintraege: int
    gespalten: int
    neben: float
    monge: bool
    permutation: bool
    eindeutig: bool
    fehler: tuple


def analysiere_instanz(inst: Instanz, regel: str, rng=None) -> Ansicht:
    loesung = loese_transport(inst.a, inst.b, inst.kosten, regel, rng)
    plan = loesung.plan
    fehler = tuple(pruefe_loesung(inst.a, inst.b, inst.kosten, loesung))
    return Ansicht(inst, loesung, eintraege(plan), gespaltene_lager(plan), nebenmasse(plan), ist_monge(plan), ist_permutation(plan),
                   ist_eindeutig(inst.kosten, plan, loesung.u, loesung.v), fehler)


def analysiere(einst: Settings) -> Ansicht:
    inst = instanz_aus_einstellungen(einst)
    return analysiere_instanz(inst, einst.regel, instanz_rng(einst.seed, C.ZWECK_ANSICHT, 99))


# ----------------------------------------------------------------------------------------------------------------------------------
# Reihen über feste Instanzen (Zähleinheit: Instanz)
# ----------------------------------------------------------------------------------------------------------------------------------

def reihe_zuordnen(familie, n, instanzen, seed, gitter=C.GITTER_STANDARD, ziehung=C.ZIEHUNGEN[0], regel=C.REGEL_REIHEN) -> dict:
    """Gleiche Massen 1/n: wie viele Simplex-Ecken sind Permutationsmatrizen, und gleicht der Wert dem der Ungarischen Methode?"""
    perm = gleich = 0
    abw = 0.0
    for i in range(instanzen):
        inst = make_instanz(familie, n, n, 0.0, seed, C.ZWECK_ZUORDNEN, i, gitter, ziehung)
        lo = loese_transport(inst.a, inst.b, inst.kosten, regel)
        perm += ist_permutation(lo.plan)
        _, wert = ungarisch(inst.kosten)
        d = abs(lo.kosten * n - wert)
        abw = max(abw, d)
        gleich += d <= 1e-9 * max(1.0, abs(wert))
    return {"familie": familie, "n": n, "instanzen": instanzen, "seed": seed, "permutation": int(perm), "gleicher_wert": int(gleich), "max_abweichung": abw}


def reihe_uebergang(familie, n, s_werte, instanzen, seed, gitter=C.GITTER_STANDARD, ziehung=C.ZIEHUNGEN[0], regel=C.REGEL_REIHEN) -> list[dict]:
    """Je Massenungleichheit s eigene Instanzen (eigener Zufallsstrom je s): gespaltene Lager, Nebenmasse, Monge-Abbildungen, volle Einträge n + m - 1."""
    zeilen = []
    for k, s in enumerate(s_werte):
        gesp, neben, monge, voll = [], [], 0, 0
        for i in range(instanzen):
            inst = make_instanz(familie, n, n, s, seed, C.ZWECK_UEBERGANG, k * 100_000 + i, gitter, ziehung)
            lo = loese_transport(inst.a, inst.b, inst.kosten, regel)
            gesp.append(gespaltene_lager(lo.plan))
            neben.append(nebenmasse(lo.plan))
            monge += ist_monge(lo.plan)
            voll += eintraege(lo.plan) == 2 * n - 1
        zeilen.append({"s": s, "familie": familie, "n": n, "instanzen": instanzen, "seed": seed,
                       "gespalten_mittel": float(np.mean(gesp)), "gespalten_min": int(min(gesp)), "gespalten_max": int(max(gesp)),
                       "neben_mittel": float(np.mean(neben)), "monge": int(monge), "voll": int(voll)})
    return zeilen


def reihe_gleichstand(familie, n, s, instanzen, seed, gitter=C.GITTER_STANDARD, ziehung=C.ZIEHUNGEN[0]) -> dict:
    """Wie viele Instanzen haben mehrere optimale Pläne (exakter Test), und liefern die drei Eintrittsregeln bei gleichem Wert verschiedene Pläne?"""
    mehrere = verschieden = 0
    abw_wert = 0.0
    spannen = []
    for i in range(instanzen):
        inst = make_instanz(familie, n, n, s, seed, C.ZWECK_GLEICHSTAND, i, gitter, ziehung)
        loesungen = [loese_transport(inst.a, inst.b, inst.kosten, r, instanz_rng(seed, C.ZWECK_GLEICHSTAND, 500_000 + i)) for r in C.REGELN]
        mehrere += not ist_eindeutig(inst.kosten, loesungen[0].plan, loesungen[0].u, loesungen[0].v)
        abw_wert = max(abw_wert, max(abs(lo.kosten - loesungen[0].kosten) for lo in loesungen))
        if not all(gleiche_plaene(loesungen[0].plan, lo.plan) for lo in loesungen[1:]):
            verschieden += 1
            zweit = [float((inst.zweit * lo.plan).sum()) for lo in loesungen]
            spannen.append((max(zweit) - min(zweit)) / max(zweit) if max(zweit) > 0 else 0.0)
    return {"familie": familie, "n": n, "s": s, "instanzen": instanzen, "seed": seed, "gitter_seite": inst.gitter_seite if instanzen else 0, "ziehung": ziehung,
            "mehrere_optima": int(mehrere), "verschiedene_plaene": int(verschieden), "max_wertabweichung": abw_wert,
            "zweit_spanne_mittel": float(np.mean(spannen)) if spannen else 0.0}


@dataclass(frozen=True)
class GleichstandAnsicht:
    inst: Instanz
    loesungen: tuple            # eine Lösung je Regel in der Reihenfolge C.REGELN
    eindeutig: bool
    zweit: tuple                # zweite Kennzahl je Regel (Summe von D · Plan)
    misch_a: int | None         # Index der beiden ersten verschiedenen Pläne (für die Mischung), sonst None
    misch_b: int | None


def gleichstand_ansicht(einst: Settings) -> GleichstandAnsicht:
    inst = make_instanz(einst.familie, einst.n, einst.n, einst.s, einst.seed, C.ZWECK_ANSICHT, 0, einst.gitter, einst.ziehung)
    loesungen = tuple(loese_transport(inst.a, inst.b, inst.kosten, r, instanz_rng(einst.seed, C.ZWECK_ANSICHT, 98)) for r in C.REGELN)
    eind = ist_eindeutig(inst.kosten, loesungen[0].plan, loesungen[0].u, loesungen[0].v)
    zweit = tuple(float((inst.zweit * lo.plan).sum()) for lo in loesungen)
    a = b = None
    for i in range(len(loesungen)):
        for j in range(i + 1, len(loesungen)):
            if a is None and not gleiche_plaene(loesungen[i].plan, loesungen[j].plan):
                a, b = i, j
    return GleichstandAnsicht(inst, loesungen, eind, zweit, a, b)


def mischung(p, q, t: float) -> np.ndarray:
    """t · P + (1 - t) · Q: liegt für jedes t zwischen 0 und 1 im Transportpolytop; bei zwei optimalen Plänen ist jede Mischung optimal."""
    return t * np.asarray(p) + (1.0 - t) * np.asarray(q)


# ----------------------------------------------------------------------------------------------------------------------------------
# Dualität: Preise und Schlupf (Handbeispiel 2)
# ----------------------------------------------------------------------------------------------------------------------------------

def preise_ansicht(inst: Instanz) -> dict:
    """Preise u, v (u_0 = 0), Schlupf c - u - v, Primal- und Dualwert und die Preisprobe: eine Einheit mehr bei Lager 3 und Kunde 1 ändert die Kosten um u_3 + v_1."""
    lo = loese_transport(inst.a, inst.b, inst.kosten, "erste")
    a2, b2 = inst.a.copy(), inst.b.copy()
    a2[-1] += 1.0
    b2[0] += 1.0
    lo2 = loese_transport(a2, b2, inst.kosten, "erste")
    return {"u": lo.u, "v": lo.v, "schlupf": inst.kosten - lo.u[:, None] - lo.v[None, :], "plan": lo.plan, "primal": lo.kosten,
            "dual": float(inst.a @ lo.u + inst.b @ lo.v), "preisprobe": lo2.kosten - lo.kosten, "preisformel": float(lo.u[-1] + lo.v[0]), "kosten_neu": lo2.kosten}


def dual_luecke(instanzen: int, seed: int, n: int = 12, familie: str = "Punkte im Einheitsquadrat", s: float = 1.0) -> dict:
    """Größte Abweichung |Primalwert - Dualwert| und größte Verletzung von u_i + v_j <= c_ij über `instanzen` Zufallsinstanzen."""
    luecke = verletzung = 0.0
    for i in range(instanzen):
        inst = make_instanz(familie, n, n, s, seed, C.ZWECK_UEBERGANG, 900_000 + i)
        lo = loese_transport(inst.a, inst.b, inst.kosten, C.REGEL_REIHEN)
        luecke = max(luecke, abs(float(inst.a @ lo.u + inst.b @ lo.v) - lo.kosten))
        verletzung = max(verletzung, float((lo.u[:, None] + lo.v[None, :] - inst.kosten).max()))
    return {"instanzen": instanzen, "n": n, "luecke": luecke, "verletzung": max(verletzung, 0.0)}


def dual_luecke_gesamt(seed: int = 2026, groessen=(3, 8, 20, 40), je_groesse: int = 25) -> dict:
    """Die Zahlenzeile zur Dualität: je Größe `je_groesse` Instanzen (Punkte im Einheitsquadrat, s = 1), größte Abweichung und größte Verletzung über alle."""
    teile = [dual_luecke(je_groesse, seed + n, n=n) for n in groessen]
    return {"instanzen": sum(t["instanzen"] for t in teile), "luecke": max(t["luecke"] for t in teile), "verletzung": max(t["verletzung"] for t in teile),
            "groessen": [t["n"] for t in teile]}


# ----------------------------------------------------------------------------------------------------------------------------------
# Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def de(x: float, stellen: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{stellen}f}".replace(".", ",")


def de_e(x: float) -> str:
    """Zahl in Exponentialschreibweise mit einer Stelle und Dezimalkomma (zum Beispiel 7,1e-15)."""
    return f"{x:.1e}".replace(".", ",")


def prozent(anteil: float, stellen: int = 1) -> str:
    return de(100.0 * anteil, stellen) + " %"


def zweit_text(r: dict) -> str:
    """Streuung der zweiten Kennzahl; ohne Instanz mit verschiedenen Plänen ist das Mittel nicht definiert, dann „–“ statt eines Nullwerts."""
    return prozent(r["zweit_spanne_mittel"]) if r["verschiedene_plaene"] > 0 else "–"


def s_text(s: float) -> str:
    return f"{s:g}".replace(".", ",")


def verdict_ansicht(a: Ansicht) -> tuple[str, str]:
    """Kernfrage 2 für die Instanz oben: wie viele Lager sind gespalten, und wie viel Masse betrifft das?"""
    n, m = a.inst.n, a.inst.m
    zaehlung = f"**{a.gespalten} von {n}** Lagern sind gespalten, **{prozent(a.neben)}** der Masse liegen außerhalb des Hauptabnehmers"
    eintraege_text = f"{a.eintraege} Einträge im Plan ({n} + {m} − 1 = {n + m - 1} bei einer Ecke ohne Entartung)"
    if a.monge:
        return "success", f"In dieser Instanz liefert jedes Lager an genau einen Kunden (Monge-Abbildung): {zaehlung}. {eintraege_text}."
    return "warning", f"In dieser Instanz ist der Plan keine Zuordnung: {zaehlung}. {eintraege_text}."


def verdict_zuordnen(r: dict) -> tuple[str, str]:
    p, k, g = r["permutation"], r["instanzen"], r["gleicher_wert"]
    kopf = f"Gleiche Massen 1/n, {r['familie']}, n = {r['n']}, {k} Instanzen:"
    wert = f"Der Wert gleicht dem der Ungarischen Methode in {g} von {k} Instanzen (größte Abweichung {de_e(r['max_abweichung'])})."
    grenze = " Gilt für Ecken des Simplex; die Mitte eines Innere-Punkte-Laufs ohne Crossover ist nicht gemessen."
    if p == k:
        return "success", f"{kopf} in **{p} von {k}** Instanzen ist die Ecke eine Permutationsmatrix. {wert}{grenze}"
    return "warning", f"{kopf} nur in **{p} von {k}** Instanzen ist die Ecke eine Permutationsmatrix. {wert}{grenze}"


def verdict_uebergang(zeilen: list[dict]) -> tuple[str, str]:
    """Zählung und Masse nebeneinander; kein Satz „bei ungleichen Massen nie“, sondern die Zählung der Reihe."""
    z0 = zeilen[0]
    n, k = z0["n"], z0["instanzen"]
    teile = [f"s = {s_text(z['s'])}: {de(z['gespalten_mittel'])} von {n} Lagern gespalten, {prozent(z['neben_mittel'], 2 if z['neben_mittel'] < 0.1 else 1)} der Masse außerhalb des Hauptabnehmers"
             for z in zeilen]
    gesamt = sum(z["monge"] for z in zeilen if z["s"] > 0)
    zahl = sum(z["instanzen"] for z in zeilen if z["s"] > 0)
    if zahl == 0:
        schluss = ""
    elif gesamt == 0:
        schluss = f" In keiner der {k} Instanzen je s > 0 liefert jedes Lager an genau einen Kunden."
    else:
        schluss = f" In {gesamt} von {zahl} Instanzen mit s > 0 liefert jedes Lager an genau einen Kunden."
    text = (f"{z0['familie']}, n = m = {n}, {k} Instanzen je s (Mittel über die Instanzen; unabhängig gezogene Massen, je s eigene Instanzen): " + "; ".join(teile) + "." + schluss)
    return "info", text


def verdict_gleichstand(r: dict) -> tuple[str, str]:
    k = r["instanzen"]
    art = "Zuordnung" if r["s"] == 0 else f"Transport (s = {s_text(r['s'])})"
    kopf = f"{art}, {r['familie']}, n = {r['n']}, {k} Instanzen:"
    ort = f" Gitterseite {r['gitter_seite']}, {r['ziehung']}." if r.get("gitter_seite") else ""
    mehr = f"in **{r['mehrere_optima']} von {k}** Instanzen gibt es mehrere optimale Pläne"
    versch = f"die drei Eintrittsregeln liefern bei gleichem Wert in {r['verschiedene_plaene']} von {k} Instanzen verschiedene Pläne"
    if r["mehrere_optima"] == 0:
        ehrlich = f" Der Plan ist in allen {k} Instanzen eindeutig; die Wahl der Regel ändert nichts."
    elif r["verschiedene_plaene"] > 0:
        ehrlich = " Es gibt keinen Fehler, nur eine Auswahl: welche Ecke herauskommt, hängt hier von der Regel ab."
    else:
        ehrlich = " Die drei Regeln wählen hier dieselbe Ecke; ein anderer Löser könnte bei Gleichstand eine andere wählen (nicht gemessen)."
    return ("info" if r["mehrere_optima"] else "success"), f"{kopf} {mehr}; {versch}.{ort}{ehrlich}"


def kurz_zuordnen(r: dict) -> str:
    return f"Bei gleichen Massen 1/n ist die Ecke in **{r['permutation']} von {r['instanzen']}** Instanzen eine Permutationsmatrix ({r['familie']}, n = {r['n']}); Experiment 1."


def kurz_uebergang(zeilen: list[dict]) -> str:
    """Eine Zeile je Kernzahl: erste und letzte Stufe von s und die kleinste Stufe über null; beide Zahlen (Zählung und Masse) stehen nebeneinander."""
    z0, zk = zeilen[0], zeilen[-1]
    klein = next((z for z in zeilen if z["s"] > 0), None)
    n = z0["n"]

    def teil(z):
        return f"s = {s_text(z['s'])}: {de(z['gespalten_mittel'])} von {n} Lagern gespalten, {prozent(z['neben_mittel'], 2 if z['neben_mittel'] < 0.1 else 1)} der Masse außerhalb des Hauptabnehmers"

    teile = [teil(z0)] + ([teil(klein)] if klein is not None and klein is not zk else []) + ([teil(zk)] if zk is not z0 else [])
    return "; ".join(teile) + f" (Mittel über {z0['instanzen']} Instanzen je s); Experiment 2."


def kurz_gleichstand(r: dict) -> str:
    return (f"In **{r['mehrere_optima']} von {r['instanzen']}** Instanzen gibt es mehrere optimale Pläne ({r['familie']}, n = {r['n']}); "
            f"die drei Eintrittsregeln wählen in {r['verschiedene_plaene']} von {r['instanzen']} Instanzen verschiedene Pläne; Experiment 3.")
