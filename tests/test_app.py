"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# ----------------------------------------------------------------------------------------------------------------------------------
# Ergänzungen: Auswahlregler, Presets mit erwarteter Verdikt-Zeile, Permalink, Randfälle (DEMO-PLAYBOOK §1d, §11a Punkt 7)
# ----------------------------------------------------------------------------------------------------------------------------------

import pytest  # noqa: E402

import ztr_constants as C  # noqa: E402
import ztr_presets as P  # noqa: E402


def _fresh_klein() -> AppTest:
    """Wie `_fresh_app`, aber mit der kleinsten Zahl von Instanzen (Adresszeile `inst=10`): die Reihen rechnen dann kaum, der Test prüft nur das Zusammenspiel der Regler."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["inst"] = "10"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(str(e.value) for kind in ("success", "info", "warning", "markdown", "caption") for e in getattr(at, kind))


@pytest.mark.parametrize("key,optionen", [("quelle", C.QUELLEN), ("familie", C.FAMILIEN), ("s_stufe", C.S_STUFEN), ("ziehung", C.ZIEHUNGEN), ("regel", C.REGELN), ("mischung", C.S_MISCH)])
def test_jede_option_der_auswahlregler_laeuft_ohne_fehler(key, optionen):
    for option in optionen:
        at = _fresh_klein()
        for art in ("selectbox", "select_slider", "radio"):
            try:
                getattr(at.sidebar, art)(key=key).set_value(option)
                break
            except KeyError:
                continue
        at.run(timeout=TIMEOUT)
        assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_gitterseiten_und_alle_gitter_familien_mit_beiden_ziehungen():
    for familie in C.GITTER_FAMILIEN:
        for ziehung in C.ZIEHUNGEN:
            for g in (C.GITTER_STANDARD, C.GITTER_MAX):
                at = _fresh_klein()
                at.sidebar.selectbox(key="familie").set_value(familie)
                at.sidebar.radio(key="ziehung").set_value(ziehung)
                at.sidebar.select_slider(key="gitter_g").set_value(g)
                at.sidebar.selectbox(key="quelle").set_value(C.QUELLE_ZUFALL)
                at.run(timeout=TIMEOUT)
                assert not at.exception, (familie, ziehung, g, [str(e) for e in at.exception])


def test_randfaelle_alle_minima_und_n_3_mit_jeder_familie():
    at = _fresh_klein()
    at.sidebar.selectbox(key="quelle").set_value(C.QUELLE_ZUFALL)
    at.sidebar.slider(key="n_slider").set_value(C.N_MIN)
    at.sidebar.slider(key="instanzen").set_value(C.INSTANZEN_MIN)
    at.sidebar.number_input(key="seed_input").set_value(0)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for familie in C.FAMILIEN:
        at.sidebar.selectbox(key="familie").set_value(familie).run(timeout=TIMEOUT)
        assert not at.exception, (familie, [str(e) for e in at.exception])


def test_preset_von_hand_zuordnung_zeigt_die_monge_abbildung_ohne_spaltung():
    at = _fresh_app()
    text = _texte(at)
    assert "Monge-Abbildung): **0 von 3** Lagern sind gespalten" in text
    assert [m.value for m in at.metric][:2] == ["5", "0 von 3"]


def test_preset_von_hand_transport_zeigt_zwei_gespaltene_lager_und_10_prozent_nebenmasse():
    at = _fresh_app()
    at.sidebar.button(key="preset_Von Hand: 3 × 3 Transport").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert [m.value for m in at.metric][:5] == ["97", "2 von 3", "10,0 %", "5 (voll: 5)", "0"]       # Gesamtkosten 97, Nebenmasse 2 von 20, die Startecke ist schon optimal
    assert "**2 von 3** Lagern sind gespalten" in _texte(at)


def test_preset_gleichstand_gitter_zeigt_mehrere_optimale_plaene_und_drei_verschiedene_plaene():
    at = _fresh_app()
    at.sidebar.button(key="preset_Gleichstand: Gitter").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = _texte(at)
    assert "Zuordnung, Gitter L1, n = 10, 100 Instanzen:" in text and "Gitterseite 6, ohne Zurücklegen." in text
    assert "keinen Fehler, nur eine Auswahl: welche Ecke herauskommt, hängt hier von der Regel ab." in text      # mehrere Optima und verschiedene Pläne: der Satz gilt
    assert not any("eindeutigen optimalen Plan" in str(i.value) for i in at.info)            # die Instanz dieses Presets hat Gleichstand
    assert at.metric[-1].label == "Ist eine Zuordnung?" and at.metric[-1].value == "nein"       # die Mitte zweier optimaler Zuordnungen ist keine Zuordnung


def test_presets_uebergang_und_viele_lager_zeigen_gespaltene_lager_und_nebenmasse():
    at = _fresh_app()
    at.sidebar.button(key="preset_Übergang s = 0,001").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = _texte(at)
    assert "Punkte im Einheitsquadrat, n = m = 20, 100 Instanzen je s" in text and "s = 0,001:" in text and "Massenungleichheit s = 0,001" in text
    assert "In keiner der 100 Instanzen je s > 0" in text
    assert "Der Plan ist in allen 100 Instanzen eindeutig; die Wahl der Regel ändert nichts." in text and "keinen Fehler, nur eine Auswahl" not in text    # Verdikt Experiment 3 ohne Gleichstand
    at.sidebar.button(key="preset_Viele Lager n = 20").click().run(timeout=TIMEOUT)
    assert not at.exception and "Massenungleichheit s = 1" in _texte(at)
    text = _texte(at)
    assert "Der Plan ist in allen 100 Instanzen eindeutig; die Wahl der Regel ändert nichts." in text and "keinen Fehler, nur eine Auswahl" not in text


def test_permalink_uebernimmt_werte_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "9999"
    at.query_params["inst"] = "5"
    at.query_params["s"] = "0.2"
    at.query_params["fam"] = "Gitter L2²"
    at.query_params["r"] = "unbekannt"
    at.query_params["t"] = "0.6"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = at.session_state
    assert s["n_slider"] == C.N_MAX and s["instanzen"] == C.INSTANZEN_MIN and s["s_stufe"] == 0.1 and s["familie"] == "Gitter L2²" and s["regel"] == C.DEFAULTS["regel"] and s["mischung"] == 0.5


def test_adresszeile_spiegelt_die_einstellungen():
    at = _fresh_app()
    at.sidebar.slider(key="instanzen").set_value(50).run(timeout=TIMEOUT)
    assert at.query_params["inst"] == ["50"] or at.query_params["inst"] == "50"


def test_die_ansicht_oben_nennt_keine_planbegriffe_und_keine_umgebungsaussagen():
    at = _fresh_app()
    gesamt = _texte(at) + " " + " ".join(e.label for e in at.expander) + " " + " ".join(s.value for s in at.subheader)
    for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Orakel", "trägt", "tragen"):
        assert verboten not in gesamt, verboten


def test_formel_expander_zeigt_preise_schlupf_dualwert_zahlenzeile_und_preisprobe():
    at = _fresh_klein()
    text = _texte(at)
    assert "Preise (mit $u_1 = 0$): $u = (0, -16, -15)$ und $v = (4, 20, 19)$. Primalwert **97**" in text and "Dualwert" in text and text.count("**97**") >= 2
    assert "In 100 Zufallsinstanzen (Punkte im Einheitsquadrat, $s = 1$, je 25 Instanzen mit $n = m$ = 3 / 8 / 20 / 40)" in text
    assert "größte Abweichung |Primalwert − Dualwert| 4,4e-16" in text and "größte Verletzung von $u_i + v_j \\le c_{ij}$ 5,6e-17" in text
    assert "$u_3 + v_1 =$ **-11**; erneutes Lösen gibt 86 statt 97, also eine Änderung um **-11**" in text
    tabelle = [df for df in at.dataframe if list(df.value.columns) == ["Lager", "K1", "K2", "K3"]]
    assert len(tabelle) == 1 and tabelle[0].value.values.tolist() == [["L1", "0", "0", "10"], ["L2", "32", "0", "26"], ["L3", "16", "0", "0"]]      # Schlupf wie von Hand gerechnet
    assert "Ausklappbereich" not in text
