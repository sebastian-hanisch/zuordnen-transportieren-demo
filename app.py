"""Vom Zuordnen zum Transportieren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Lager mit Angeboten, Kunden mit Nachfragen, Kosten je Einheit: der billigste Transportplan, gelöst mit einem eigenen Transportsimplex (numpy).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ztr_constants as C
import ztr_evaluation as E
from ztr_layout import dunkles_design, kopf_css
from ztr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ztr_scenario import handbeispiel_transport
from ztr_visualization import build_plan_karte, build_plan_matrix, build_uebergang

st.set_page_config(page_title="Vom Zuordnen zum Transportieren – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _ansicht(settings):
    return E.analysiere(settings)


@st.cache_data(show_spinner=False)
def _reihe_zuordnen(familie, n, instanzen, seed, gitter, ziehung):
    return E.reihe_zuordnen(familie, n, instanzen, seed, gitter, ziehung)


@st.cache_data(show_spinner=False)
def _reihe_uebergang(familie, n, instanzen, seed, gitter, ziehung):
    return E.reihe_uebergang(familie, n, C.S_STUFEN, instanzen, seed, gitter, ziehung)


@st.cache_data(show_spinner=False)
def _reihe_gleichstand(familie, n, s, instanzen, seed, gitter, ziehung):
    return E.reihe_gleichstand(familie, n, s, instanzen, seed, gitter, ziehung)


@st.cache_data(show_spinner=False)
def _gleichstand_ansicht(settings):
    return E.gleichstand_ansicht(settings)


@st.cache_data(show_spinner=False)
def _preise():
    return E.preise_ansicht(handbeispiel_transport())


@st.cache_data(show_spinner=False)
def _dual_zeile():
    return E.dual_luecke_gesamt()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ztr_layout.py).
st.html(kopf_css(dunkles_design()))
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')   # breite Tabellen scrollen auf dem Telefon
with st.container(key="kopf"):
    st.title("🚚 Vom Zuordnen zum Transportieren")
st.markdown(
    """
Ein **Transportplan** (englisch transport plan) sagt, welches **Lager** wie viel an welchen **Kunden** liefert, bei kleinsten Gesamtkosten. Sind alle Mengen gleich groß, ist die
Antwort bei gleich vielen Lagern und Kunden eine **Zuordnung** (als Ecke gibt es immer eine optimale): jedes Lager liefert an genau einen Kunden. Bei verschiedenen Mengen **kann** der Plan Lager auf mehrere
Kunden **spalten**; bei den zufällig gezogenen Massen dieser Demo tut er es in allen gemessenen Instanzen. Die Demo **zählt**, wie viele Lager
das betrifft und **wie viel Masse**, und zeigt, dass bei Gleichstand der gezeigte Plan nur **eine Auswahl** unter gleich guten ist. Gerechnet wird mit einem eigenen
**Transportsimplex** (Verfahren über Ecken), dessen Ergebnisse die Tests gegen unabhängige Verfahren prüfen.

Weiter unten: der Expander **💬 In einfachen Worten**, **❓ Kernfragen**, die drei **🔬 Experimente**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und
**📐 Mathematische Formulierung** (mit den Preisen, englisch dual variables).
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/transportplan-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein **Transportplan** sagt, welches Lager wie viel an welchen Kunden liefert, bei kleinsten Gesamtkosten. Haben alle Lager und alle Kunden gleich viel, gibt es immer einen billigsten Plan, in dem jedes Lager an genau einen Kunden liefert: eine **Zuordnung**. Bei verschiedenen Mengen **kann** der Plan Lager auf mehrere Kunden **spalten**; bei den zufällig gezogenen Mengen dieser Demo tut er es in allen gemessenen Beispielen. Die Demo zählt, wie viele Lager gespalten sind, und wiegt, wie viel Menge das betrifft: Die Zählung springt schon bei kleinen Mengenunterschieden, die betroffene Menge wächst stetig. Gibt es mehrere gleich gute Pläne, zeigt die Demo nur **eine Auswahl** davon; ein anderes Verfahren kann einen anderen wählen. **Preise** für Lager und Kunden prüfen den Plan: Passen sie dazu, gibt es keinen billigeren.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/transportplan-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert der Transportsimplex", expanded=True):
    st.markdown(
        """
1. **Aufgabe.** $n$ Lager mit Angeboten $a_i$, $m$ Kunden mit Nachfragen $b_j$ (beide Summen gleich), Kosten $c_{ij}$ je Einheit. Gesucht ist ein Plan $x_{ij} \\ge 0$ mit den
   **Randverteilungen** (englisch marginals) $a$ und $b$ als Zeilen- und Spaltensummen und kleinsten Gesamtkosten.
2. **Ecken.** Das Optimum liegt in einer Ecke des Planraums (Transportpolytop). Eine nicht entartete Ecke hat genau $n + m - 1$ Einträge ungleich null; die Zellen dieser
   Einträge bilden einen Baum zwischen Lagern und Kunden.
3. **Preise.** Aus dem Baum folgen Preise $u_i$ für die Lager und $v_j$ für die Kunden mit $u_i + v_j = c_{ij}$ auf den Baumzellen. Eine Zelle mit
   $c_{ij} - u_i - v_j < 0$ (reduzierte Kosten) lohnt sich: Der Plan wird auf dem Kreis, den sie im Baum schließt, verschoben, bis eine Zelle leer wird.
4. **Eintrittsregel.** Gibt es mehrere lohnende Zellen, wählt eine Regel (erste, letzte oder zufällige): bei Gleichständen **kann sie entscheiden, welche optimale Ecke** herauskommt (in den gemessenen Reihen meist, nicht immer).
5. **Zuordnung als Sonderfall.** Bei gleichen Massen $1/n$ sind alle Ecken Permutationsmatrizen (Satz von Birkhoff); **bei verschiedenen Mengen kann** der Plan Lager **spalten** (bei zufällig gezogenen Massen in allen gemessenen Instanzen). „Lager spalten“
   heißt hier: eine Zeile des Plans hat mehr als einen Eintrag. Ein Plan ohne gespaltenes Lager ist eine **Monge-Abbildung** (englisch Monge map).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧩 Instanz")
    st.selectbox("Instanz oben", C.QUELLEN, key="quelle",
                 help="Die Ansicht oben zeigt ein Handbeispiel oder eine Zufallsinstanz. Die übrigen Regler dieser Gruppe steuern die Zufallsinstanz und die Experimente.")
    st.slider("Lager = Kunden (n)", *bounds("n_slider"), key="n_slider", help="Zahl der Lager; ebenso viele Kunden. Für die Handbeispiele ohne Wirkung, für die Experimente immer wirksam.")
    st.select_slider("Massenungleichheit s", options=C.S_STUFEN, key="s_stufe", format_func=E.s_text,
                     help="Massen a = (1 − s)/n + s · Dirichlet(1), ebenso b. s = 0: alle Massen gleich 1/n. s = 1: unabhängig gezogene Dirichlet(1)-Massen.")
    st.selectbox("Kostenfamilie", C.FAMILIEN, key="familie", help="Wie die Kostenmatrix entsteht; Beschreibung unter den Experimenten.")
    gitter = st.session_state.get("familie") in C.GITTER_FAMILIEN
    st.select_slider("Gitterseite g", options=(C.GITTER_STANDARD,) + tuple(range(C.GITTER_MIN, C.GITTER_MAX + 1)), key="gitter_g",
                     format_func=lambda g: "Standard" if g == C.GITTER_STANDARD else str(g), disabled=not gitter,
                     help="Seitenlänge des Gitters (nur bei den Gitter-Familien). Standard: aufgerundete Wurzel aus n + m, plus 1. Ohne Zurücklegen ist die Seite mindestens so groß, "
                          "dass n + m verschiedene Punkte Platz haben.")
    st.radio("Ziehung der Gitterpunkte", C.ZIEHUNGEN, key="ziehung", disabled=not gitter,
             help="Ohne Zurücklegen sind alle Orte verschieden, mit Zurücklegen können Orte doppelt vorkommen (nur bei den Gitter-Familien).")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.header("🔬 Experimente")
    st.slider("Zahl der Instanzen", *bounds("instanzen"), key="instanzen", step=C.INSTANZEN_STEP, help="So viele Instanzen je Reihe in den drei Experimenten (Zähleinheit ist immer die Instanz).")
    st.radio("Eintrittsregel des Simplex", C.REGELN, key="regel", horizontal=True,
             help="Welche Zelle mit negativen reduzierten Kosten als nächste in die Basis kommt. Wirkt auf die Instanz oben und bei Gleichstand auf die Wahl der Ecke.")
    st.select_slider("Mischung t (Experiment 3)", options=C.S_MISCH, key="mischung", format_func=E.s_text,
                     help="Anteil des ersten von zwei optimalen Plänen in der Mischung t · P₁ + (1 − t) · P₂.")

einst = E.settings_aus_zustand(st.session_state)
sync_query_params({k: st.session_state[k] for k in C.DEFAULTS})

# --- Ergebnis: eine Instanz ----------------------------------------------------------------------------------------------------------------

a = _ansicht(einst)
inst, lo = a.inst, a.loesung
st.markdown("## 🎯 Der Plan zu einer Instanz")
if einst.quelle == C.QUELLE_ZUFALL:
    gitter_text = f", Gitterseite {inst.gitter_seite} ({einst.ziehung})" if inst.gitter_seite else ""
    st.caption(f"{inst.n} Lager, {inst.m} Kunden, {inst.familie}{gitter_text}, Massenungleichheit s = {E.s_text(einst.s)}, Zufalls-Seed {einst.seed}, Eintrittsregel „{einst.regel}“.")
else:
    st.caption(f"{einst.quelle}: {inst.n} Lager, {inst.m} Kunden, feste Zahlen (n, s, Kostenfamilie und Zufalls-Seed gelten nur für die Zufallsinstanz und die Experimente).")
if a.fehler:
    st.error("Die Prüfung des Ergebnisses schlägt an: " + "; ".join(a.fehler))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Gesamtkosten", f"{lo.kosten:.4g}".replace(".", ","), help="Summe aus Kosten mal Menge über alle Zellen des Plans.")
m2.metric("Gespaltene Lager", f"{a.gespalten} von {inst.n}", help="Zeilen des Plans mit mehr als einem Eintrag.")
m3.metric("Nebenmasse", E.prozent(a.neben), help="Anteil der Gesamtmasse, den ein Lager nicht an seinen größten Abnehmer liefert.")
m4.metric("Einträge im Plan", f"{a.eintraege} (voll: {inst.n + inst.m - 1})", help="Eine nicht entartete Ecke hat n + m − 1 Einträge; weniger heißt entartet (etwa bei gleichen Massen).")
m5.metric("Simplex-Schritte", f"{lo.schritte}", help=f"Davon entartet (Schrittweite 0): {lo.entartete_schritte}.")
kind, text = E.verdict_ansicht(a)
getattr(st, kind)(text)
spalte_links, spalte_rechts = st.columns(2)
with spalte_links:
    st.plotly_chart(build_plan_matrix(lo.plan, inst.a, inst.b), width="stretch", key="main_matrix")
    st.caption("Plan als Matrix: Zeilen sind Lager (mit Angebot), Spalten Kunden (mit Nachfrage); leere Zellen haben Menge 0.")
with spalte_rechts:
    if inst.lager is not None:
        st.plotly_chart(build_plan_karte(inst.lager, inst.kunden, lo.plan), width="stretch", key="main_karte")
        st.caption("Plan als Pfeile auf der Karte (Lager = Quadrate, Kunden = Kreise); die Breite des Pfeils wächst mit der Menge.")
    else:
        st.info("Diese Instanz hat keine Orte (Kosten ohne Geometrie): der Plan steht nur als Matrix links.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

gitter_arg = (einst.gitter, einst.ziehung)
with st.spinner("Rechne die drei Reihen (je Instanz ein Simplex-Lauf) …"):
    r1 = _reihe_zuordnen(einst.familie, einst.n, einst.instanzen, einst.seed, *gitter_arg)
    r2 = _reihe_uebergang(einst.familie, einst.n, einst.instanzen, einst.seed, *gitter_arg)
    r3 = _reihe_gleichstand(einst.familie, einst.n, einst.s, einst.instanzen, einst.seed, *gitter_arg)

st.markdown("---")
st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wann ist die Antwort eine Zuordnung?**")
st.markdown(E.kurz_zuordnen(r1))
st.markdown("**2. Wie viele Lager spaltet der Plan, und wie viel Masse betrifft das?**")
st.markdown(E.kurz_uebergang(r2))
st.markdown("**3. Welcher von mehreren gleich guten Plänen wird gezeigt?**")
st.markdown(E.kurz_gleichstand(r3))
st.caption("Alle drei Antworten rechnen live mit den Einstellungen der Seitenleiste (Kostenfamilie, n, Massenungleichheit s, Gitterseite, Ziehung, Zahl der Instanzen, Zufalls-Seed; die erste und die zweite Antwort hängen nicht von s ab); die Verdikte mit den Einschränkungen stehen in den Experimenten.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Zuordnen bei gleichen Massen")
st.caption(f"Alle Massen gleich 1/n (s = 0), {einst.familie}, n = {einst.n}, {einst.instanzen} Instanzen, Zufalls-Seed {einst.seed}. Vergleich: Ungarische Methode (Kuhn-Munkres), "
           "ein eigenes, von der Simplex-Rechnung unabhängiges Verfahren für die Zuordnung. Die Reihe rechnet mit der Eintrittsregel „steilste“ (größte Abnahme der reduzierten Kosten).")
e1, e2, e3 = st.columns(3)
e1.metric("Ecke ist Permutationsmatrix", f"{r1['permutation']} von {r1['instanzen']}")
e2.metric("Wert wie Ungarische Methode", f"{r1['gleicher_wert']} von {r1['instanzen']}")
e3.metric("Größte Wertabweichung", E.de_e(r1['max_abweichung']))
kind, text = E.verdict_zuordnen(r1)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Massen ungleich machen, Lager spalten")
st.caption(f"{einst.familie}, n = m = {einst.n}, je Massenungleichheit s eigene {einst.instanzen} Instanzen (unabhängig gezogen), Zufalls-Seed {einst.seed}; "
           "Mittel über die Instanzen, im linken Diagramm Balken von kleinstem bis größtem Wert. Die Reihe rechnet mit der Eintrittsregel „steilste“.")
tabelle = [
    {
        "s": E.s_text(z["s"]),
        "gespaltene Lager (Mittel [Min, Max])": f"{E.de(z['gespalten_mittel'], 2)} [{z['gespalten_min']}, {z['gespalten_max']}] von {z['n']}",
        "Nebenmasse (Mittel)": E.prozent(z["neben_mittel"], 3 if z["neben_mittel"] < 0.01 else 2),
        "Monge-Abbildung": f"{z['monge']} von {z['instanzen']}",
        f"Einträge = n + m − 1 = {2 * z['n'] - 1}": f"{z['voll']} von {z['instanzen']}",
    }
    for z in r2
]
st.dataframe(tabelle, hide_index=True, width="stretch")
fig_zahl, fig_masse = build_uebergang(r2, E.s_text)
c_links, c_rechts = st.columns(2)
c_links.plotly_chart(fig_zahl, width="stretch", key="uebergang_zahl")
c_rechts.plotly_chart(fig_masse, width="stretch", key="uebergang_masse")
kind, text = E.verdict_uebergang(r2)
getattr(st, kind)(text)
st.caption("Zählweise zählt: Nach der **Zahl gespaltener Lager** springt der Befund beim Schritt von s = 0 zu s = 0,001, nach der **Masse** wächst er stetig mit s.")

st.markdown("---")
st.subheader("🔬 Experiment 3: Welcher Plan? Gleichstände")
art = "Zuordnung (alle Massen gleich)" if einst.s == 0 else f"Transport mit Massenungleichheit s = {E.s_text(einst.s)}"
ort = f", Gitterseite {r3['gitter_seite']} ({einst.ziehung})" if r3.get("gitter_seite") else ""
st.caption(f"{art}, {einst.familie}{ort}, n = {einst.n}, {einst.instanzen} Instanzen, Zufalls-Seed {einst.seed}. Drei Eintrittsregeln desselben Simplex, Eindeutigkeitstest je Instanz (exakt bei ganzzahligen und Gitterkosten, bei stetigen Kosten mit der Toleranz 1e-9 auf den reduzierten Kosten).")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Mehrere optimale Pläne", f"{r3['mehrere_optima']} von {r3['instanzen']}")
g2.metric("Regeln wählen verschiedene Pläne", f"{r3['verschiedene_plaene']} von {r3['instanzen']}")
g3.metric("Größte Wertabweichung", E.de_e(r3['max_wertabweichung']), help="Zwischen den Plänen der drei Regeln; der Wert ist derselbe, der Plan kann verschieden sein.")
g4.metric("Zweite Kennzahl streut um", E.zweit_text(r3),
          help="Mittel über die Instanzen mit verschiedenen Plänen: (größte − kleinste) / größte Summe aus zweiter Kostenmatrix mal Plan, über die drei Pläne. „–“: keine Instanz mit verschiedenen Plänen, das Mittel ist dann nicht definiert.")
kind, text = E.verdict_gleichstand(r3)
getattr(st, kind)(text)
ga = _gleichstand_ansicht(einst)
if ga.eindeutig:
    st.info("Die Instanz mit dem Zufalls-Seed der Seitenleiste hat einen eindeutigen optimalen Plan: alle drei Regeln liefern denselben. Anderen Zufalls-Seed oder eine Familie mit "
            "Gleichständen (Gitter, ganzzahlig) wählen.")
spalten = st.columns(3)
for sp, regel, loesung, zweit in zip(spalten, C.REGELN, ga.loesungen, ga.zweit):
    with sp:
        titel = f"Regel „{regel}“: Kosten {loesung.kosten:.4g}".replace(".", ",") + f" · zweite Kennzahl {zweit:.4g}".replace(".", ",")
        st.plotly_chart(build_plan_matrix(loesung.plan, ga.inst.a, ga.inst.b, titel=titel, hoehe=330, beschriftung=False), width="stretch", key=f"gleich_{regel}")
st.caption("Die zweite Kennzahl ist die Summe aus einer zweiten Kostenmatrix mal Plan (Gitter: das jeweils andere Abstandsmaß; sonst eine unabhängig gezogene zweite Matrix gleicher Art). "
           "Gleich teure Pläne können nach ihr verschieden sein.")
if ga.misch_a is not None:
    pa, pb = ga.loesungen[ga.misch_a], ga.loesungen[ga.misch_b]
    gemischt = E.mischung(pa.plan, pb.plan, einst.mischung)
    kosten_misch = float((ga.inst.kosten * gemischt).sum())
    st.markdown(f"**Mischung** t · Plan „{C.REGELN[ga.misch_a]}“ + (1 − t) · Plan „{C.REGELN[ga.misch_b]}“ mit t = {E.s_text(einst.mischung)}")
    mm1, mm2, mm3 = st.columns(3)
    mm1.metric("Kosten der Mischung", f"{kosten_misch:.4g}".replace(".", ","), help=f"Kosten der beiden Ausgangspläne: {pa.kosten:.4g} und {pb.kosten:.4g}.".replace(".", ","))
    mm2.metric("Einträge der Mischung", f"{E.eintraege(gemischt)}")
    mm3.metric("Ist eine Zuordnung?", "ja" if E.ist_permutation(gemischt) else "nein")
    st.plotly_chart(build_plan_matrix(gemischt, ga.inst.a, ga.inst.b, hoehe=330, beschriftung=False), width="stretch", key="gleich_mischung")
    st.caption("Jede Mischung zweier optimaler Pläne ist zulässig und kostet dasselbe; die Mitte (t = 0,5) ist bei einer Zuordnung keine Zuordnung mehr.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme dieser Demo | Was passiert, wenn sie nicht gilt | Wo es weitergeht |
|---|---|---|
| **Der optimale Plan ist eine dünne Ecke (höchstens n + m − 1 Einträge), und bei Gleichstand wählt der Löser eine.** | Ein weicher Plan (mit Entropie-Regularisierung) ist dicht und eindeutig und wählt bei Gleichstand (für kleines ε) den optimalen Plan mit größter Entropie, also einen Punkt im Inneren der Fläche. | Sinkhorn-Verfahren, weiche Pläne |
| **Minimum bei festen Randsummen.** Der billigste Plan, wenn Angebote und Nachfragen vorgegeben sind. | Wer eine alte Matrix an neue Randsummen anpasst, sucht die Matrix mit kleinster Kullback-Leibler-Abweichung (englisch KL divergence) zur alten, nicht die billigste (Projektion statt Minimum). | Matrix abgleichen (Furness, IPF) |
| **Lager, Kunden und Kosten sind gegeben; der Optimalwert ist das Ende der Rechnung, ein Preis für einen Auftrag.** | Als Abstand zwischen zwei Verteilungen hängt der Wert von Raster und Kostenwahl ab. | Abstände zwischen Verteilungen |
| **Angebot und Nachfrage sind in der Summe gleich.** | Der Löser lehnt ungleiche Summen mit einer Fehlermeldung ab, die App bietet sie nicht an. Teilbedienung, Strafen und Ausreißer ändern die Aufgabe. | Angebot ungleich Nachfrage |
| **Bei Gleichstand ist die Wahl des optimalen Plans für den Wert gleichgültig.** | Für den Wert stimmt das, für ein zweites Kriterium nicht (Experiment 3), und eine Zwischenverteilung kann von der Wahl abhängen (hier nicht gerechnet). | Zwischen Verteilungen |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Ausgeglichene Massen (Summe der Angebote gleich Summe der Nachfragen), Gesamtmenge 1 bei den Zufallsinstanzen; Kosten je Einheit, keine Fixkosten, keine Kapazitäten auf den Wegen.
- Der Plan ist eine Ecke des Transportpolytops (Ergebnis des Simplex); gezählt wird je Instanz.
- Zufallsinstanzen entstehen aus dem Hausgenerator SplitMix64 mit dem Zufalls-Seed der Seitenleiste; jede Reihe nennt Kostenfamilie, n, Zahl der Instanzen und Zufalls-Seed.

**Grenzen.**
- Die Zählungen gelten für diese Kostenfamilien, diese Zufallszüge für die Massen und diese Zähleinheit (Instanz); „bei ungleichen Massen nie eine Zuordnung“ wird **nicht** behauptet:
  Sind die Massen auf beiden Seiten gleich verteilt oder passend vertauscht, kann das Optimum eine Zuordnung ohne Spaltung sein (Gegenbeispiele stehen in der README und in den Tests).
- Ein Innere-Punkte-Verfahren ohne Crossover (Ergebnis in der Mitte der optimalen Fläche statt in einer Ecke) ist nicht gemessen.
- Welche optimale Ecke ein Löser bei Gleichstand wählt, hängt vom Verfahren, der Regel und der Reihenfolge der Zellen ab; andere Löser und andere Versionen können andere Pläne liefern.
- Der Rechenaufwand wird nicht verglichen (kein Vergleich mit anderen Verfahren nach Schritten oder Zeit).

**Bekannte Probleme und Unsicherheiten.**
- Im Browser ist die Seite nur stichprobenartig mit echten Mausklicks durchgespielt (Presets, Scrollen); die automatischen Tests laufen über die Streamlit-Testumgebung.
- Bei großem n und vielen Instanzen rechnen die drei Reihen spürbar länger (Hinweis „Rechne die drei Reihen“); danach stehen die Ergebnisse im Zwischenspeicher.
- Hinweis zu den Gitter-Familien: Die Zahl der Instanzen mit mehreren optimalen Plänen hängt stark an Gitterseite und Ziehung der Punkte; deshalb sind beide Regler.
        """
    )

# --- Formeln -------------------------------------------------------------------------------------------------------------------------------

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/transportplan-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Angebote $a_i$ der Lager, die Nachfragen $b_j$ der Kunden und die Kosten $c_{ij}$ je Einheit. Gesucht ist ein Plan $x_{ij}$, der die Gesamtkosten minimiert. Das Modell ist vollständig: alle Variablen, Nebenbedingungen und die Gleichheit der Summen stehen unten.

| Symbol | Bedeutung |
|---|---|
| $n$, $m$ | Zahl der Lager, Zahl der Kunden |
| $i = 1, \dots, n$ | Lager (Zeilen des Plans) |
| $j = 1, \dots, m$ | Kunden (Spalten des Plans) |
| $a_i \ge 0$, $b_j \ge 0$ | Angebot des Lagers $i$, Nachfrage des Kunden $j$, mit $\sum_i a_i = \sum_j b_j$ |
| $c_{ij}$ | Kosten je Einheit von Lager $i$ zu Kunde $j$ |
| $x_{ij} \ge 0$ | Menge von Lager $i$ an Kunde $j$ (der Plan) |
| $u_i$, $v_j$ | Preise (Kantorovich-Potentiale, englisch dual variables); bei einem nicht entarteten Plan bis auf eine Konstante bestimmt |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/transportplan-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Primales Problem (Transportproblem).**
$$\min_{x \ge 0} \; \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij}\, x_{ij}$$
$$\text{so dass} \quad \sum_{j=1}^{m} x_{ij} = a_i \;\; (i = 1, \dots, n), \qquad \sum_{i=1}^{n} x_{ij} = b_j \;\; (j = 1, \dots, m).$$

**Ecken.** Jede Ecke des zulässigen Bereichs hat höchstens $n + m - 1$ Einträge $x_{ij} > 0$ (eine Randbedingung ist überflüssig, weil die Summen gleich sind). Bei gleichen
Massen $a_i = b_j = 1/n$ ist jede Ecke eine Permutationsmatrix mal $1/n$ (Satz von Birkhoff); die Aufgabe ist dann die Zuordnung. Ein Lager heißt **gespalten**, wenn seine Zeile mehr als einen
Eintrag hat. Bei einer nicht entarteten Ecke gilt $\sum_i (k_i - 1) = m - 1$, wenn $k_i$ die Zahl der Einträge der Zeile $i$ ist (denn $\sum_i k_i = n + m - 1$).

**Duales Problem (Preise).**
$$\max_{u, v} \; \sum_{i=1}^{n} a_i\, u_i + \sum_{j=1}^{m} b_j\, v_j \qquad \text{so dass} \qquad u_i + v_j \le c_{ij} \;\; \text{für alle } i, j.$$
Im Optimum sind beide Werte gleich (starke Dualität), und ein Eintrag $x_{ij} > 0$ ist nur dort möglich, wo $c_{ij} - u_i - v_j = 0$ gilt (komplementärer Schlupf). Mit $(u, v)$ ist für jede Konstante $\delta$ auch $(u + \delta,\; v - \delta)$ zulässig und optimal, und bei einem nicht entarteten Plan sind die Preise bis auf diese Konstante bestimmt; bei Entartung (etwa bei gleichen Massen) gibt es weitere optimale Preise.

**Eintrittsregeln und Gleichstand.** Die optimale Menge besteht aus allen zulässigen Plänen, die nur auf Zellen mit $c_{ij} - u_i - v_j = 0$ liegen. Hat sie mehr als einen Punkt, ist
auch jede Mischung $t P_1 + (1 - t) P_2$ ($0 \le t \le 1$) optimal; der Simplex liefert eine Ecke dieser Menge, und welche, hängt von der Eintrittsregel ab.
        """
    )
    st.markdown("**Handbeispiel von Hand nachgerechnet: Preise lesen.** Lager $(0,0)$, $(4,0)$, $(2,3)$ mit Angeboten $9, 6, 5$; Kunden $(0,2)$, $(4,2)$, $(2,5)$ mit Nachfragen $8, 8, 4$; Kosten sind quadrierte Abstände.")
    p = _preise()
    st.markdown(
        f"Preise (mit $u_1 = 0$): $u = ({', '.join(f'{x:.0f}' for x in p['u'])})$ und $v = ({', '.join(f'{x:.0f}' for x in p['v'])})$. "
        f"Primalwert **{p['primal']:.0f}**, Dualwert $\\sum a_i u_i + \\sum b_j v_j =$ **{p['dual']:.0f}**."
    )
    st.dataframe(
        [{"Lager": f"L{i + 1}", **{f"K{j + 1}": f"{p['schlupf'][i, j]:.0f}" for j in range(3)}} for i in range(3)],
        hide_index=True, width="content",
    )
    st.caption("Schlupf $c_{ij} - u_i - v_j$: null genau auf den Zellen, die der optimale Plan benutzt (komplementärer Schlupf), sonst positiv.")
    dz = _dual_zeile()
    st.markdown(
        f"**Zahlenzeile.** In {dz['instanzen']} Zufallsinstanzen (Punkte im Einheitsquadrat, $s = 1$, je 25 Instanzen mit $n = m$ = {' / '.join(str(g) for g in dz['groessen'])}): "
        f"größte Abweichung |Primalwert − Dualwert| {E.de_e(dz['luecke'])}, größte Verletzung von $u_i + v_j \\le c_{{ij}}$ {E.de_e(dz['verletzung'])} (Rundung; Selbstprobe mit den Preisen des eigenen Simplex, die unabhängige Gegenprobe steht in den Tests)."
    )
    st.markdown(
        f"**Preisprobe.** Eine Einheit mehr bei Lager 3 und bei Kunde 1 ändert die Kosten um $u_3 + v_1 =$ **{p['preisformel']:.0f}**; erneutes Lösen gibt {p['kosten_neu']:.0f} statt {p['primal']:.0f}, "
        f"also eine Änderung um **{p['preisprobe']:.0f}**. Bereiche, in denen ein Preis gilt, und Entartung zeigt die Demo „Dualität und Sensitivität“ des Portfolios."
    )
    st.markdown(
        r"""
**Literatur.** Der Satz von Birkhoff und die Dualität des Transportproblems stehen in Lehrbüchern zur Linearen Optimierung; die Demo belegt sie nur an Beispielen und durch den
Vergleich mit unabhängigen Verfahren in den Tests, nicht durch Nachlesen von Originalquellen (Quelle nicht eingesehen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html)."
)
