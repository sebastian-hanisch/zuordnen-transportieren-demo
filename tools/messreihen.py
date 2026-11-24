"""Rechnet die Messreihen der README mit dem Code der Demo neu und schreibt `tools/messreihen.json` (Aufruf aus dem Demo-Ordner:
`python tools/messreihen.py`; dauert einige Minuten). `tests/test_claims.py` prüft jede Zahl der README gegen diese Datei und rechnet die kleinen Reihen neu.

Zähleinheit überall: die Instanz. Zufall: SplitMix64 (`ztr_rng.py`), Zufalls-Seed MESS_SEED; je Zweck und Index ein eigener Zufallsstrom."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ztr_constants as C  # noqa: E402
import ztr_evaluation as E  # noqa: E402

MESS_SEED = 2026
S_UEBERGANG = (0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0)
PUNKTE = "Punkte im Einheitsquadrat"
STETIG = "stetig (Zufallskosten)"


def main() -> None:
    t0 = time.time()
    aus: dict = {"meta": {"seed": MESS_SEED, "regel_reihen": C.REGEL_REIHEN}}

    # A. Gleiche Massen 1/n: ist die Ecke eine Permutationsmatrix, und stimmt der Wert mit der Ungarischen Methode?
    aus["zuordnen"] = [E.reihe_zuordnen(fam, n, N, MESS_SEED) for fam in C.FAMILIEN for n, N in ((5, 200), (10, 200), (20, 100), (40, 20))]
    print("A fertig", round(time.time() - t0), "s", flush=True)

    # B. Übergang von der Zuordnung zum Transport (n = m = 20, Punkte im Einheitsquadrat, je s eigene Instanzen), Zusatzreihen
    aus["uebergang"] = E.reihe_uebergang(PUNKTE, 20, S_UEBERGANG, 100, MESS_SEED)
    aus["uebergang_1000"] = E.reihe_uebergang(PUNKTE, 20, (0.001, 0.03), 1000, MESS_SEED + 1)
    aus["uebergang_stetig"] = E.reihe_uebergang(STETIG, 20, (1.0,), 200, MESS_SEED)
    aus["uebergang_n"] = [E.reihe_uebergang(PUNKTE, n, (1.0,), 60, MESS_SEED)[0] for n in (5, 10, 20, 40)]
    print("B fertig", round(time.time() - t0), "s", flush=True)

    # C. Gleichstände: Zuordnung n = 10 (Standardgitter ohne Zurücklegen), Gitterseiten mit Zurücklegen, Transport
    aus["gleichstand"] = [E.reihe_gleichstand(fam, 10, 0.0, 300, MESS_SEED) for fam in (STETIG, "ganzzahlig 0 bis 10", "ganzzahlig 0 bis 2", "Gitter L1", "Gitter L2²")]
    aus["gleichstand_gitter"] = [E.reihe_gleichstand(fam, 10, 0.0, 150, MESS_SEED, g, C.ZIEHUNGEN[1]) for fam in C.GITTER_FAMILIEN for g in (6, 10, 20)]
    aus["gleichstand_transport"] = [E.reihe_gleichstand(fam, 15, 1.0, 150, MESS_SEED) for fam in (STETIG, "Gitter L1")]
    print("C fertig", round(time.time() - t0), "s", flush=True)

    # D. Dualität: größte Abweichung Primal gegen Dual und größte Verletzung von u_i + v_j <= c_ij über 100 Instanzen (vier Größen zu je 25)
    aus["dualitaet"] = E.dual_luecke_gesamt(MESS_SEED)
    print("D fertig", round(time.time() - t0), "s", flush=True)

    # E. Zahlen der Presets (Hilfetexte und Kernfragen): je Preset die Reihen mit den Einstellungen des Presets
    aus["presets"] = {}
    for name, p in C.PRESETS.items():
        s = E.settings_aus_zustand(p)
        aus["presets"][name] = {"zuordnen": E.reihe_zuordnen(s.familie, s.n, s.instanzen, s.seed, s.gitter, s.ziehung),
                                "uebergang": E.reihe_uebergang(s.familie, s.n, C.S_STUFEN, s.instanzen, s.seed, s.gitter, s.ziehung),
                                "gleichstand": E.reihe_gleichstand(s.familie, s.n, s.s, s.instanzen, s.seed, s.gitter, s.ziehung)}
    print("E fertig", round(time.time() - t0), "s", flush=True)

    (ROOT / "tools" / "messreihen.json").write_text(json.dumps(aus, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("geschrieben:", ROOT / "tools" / "messreihen.json", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
