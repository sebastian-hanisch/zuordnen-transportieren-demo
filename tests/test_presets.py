"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzen und einrasten."""

import pytest

import ztr_constants as C
import ztr_evaluation as E
import ztr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.SETTING_SPECS), name          # jedes Preset legt ALLE Regler fest
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_default_preset_equals_the_default_settings():
    erstes = next(iter(C.PRESETS))
    assert C.PRESETS[erstes] == C.DEFAULTS
    assert E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


def test_setting_specs_cover_every_preset_key_and_every_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(C.DEFAULTS)


def test_settings_aus_zustand_nimmt_die_werte_und_faellt_auf_die_standardwerte_zurueck():
    s = E.settings_aus_zustand({"n_slider": 7, "s_stufe": 0.01, "familie": "Gitter L2²", "regel": "letzte"})
    assert (s.n, s.s, s.familie, s.regel, s.seed) == (7, 0.01, "Gitter L2²", "letzte", C.DEFAULTS["seed_input"])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laesst_sich_analysieren(name):
    p = C.PRESETS[name]
    a = E.analysiere(E.settings_aus_zustand(p))
    assert a.fehler == ()


def test_einrasten_und_begrenzen_des_permalinks():
    sp = P.SETTING_SPECS
    assert P._eingerastet(sp["n_slider"], 500) == C.N_MAX and P._eingerastet(sp["n_slider"], -3) == C.N_MIN
    assert P._eingerastet(sp["s_stufe"], 0.2) == 0.1 and P._eingerastet(sp["s_stufe"], 0.0004) == 0.0 and P._eingerastet(sp["s_stufe"], 7.0) == 1.0
    assert P._eingerastet(sp["gitter_g"], 3) == C.GITTER_MIN and P._eingerastet(sp["gitter_g"], 1) == C.GITTER_STANDARD
    assert P._eingerastet(sp["familie"], "Gitter L1") == "Gitter L1" and P._eingerastet(sp["familie"], "unbekannt") is None
    assert P._eingerastet(sp["mischung"], 0.6) == 0.5 and P._eingerastet(sp["instanzen"], 5) == C.INSTANZEN_MIN
    assert P._eingerastet(sp["instanzen"], 57) == 60 and P._eingerastet(sp["instanzen"], 94) == 90 and P._eingerastet(sp["instanzen"], 999) == C.INSTANZEN_MAX
