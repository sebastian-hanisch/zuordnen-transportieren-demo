"""Gemeinsame Fixtures: die Handbeispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ztr_scenario import handbeispiel_transport, handbeispiel_zuordnung  # noqa: E402


@pytest.fixture
def hand1():
    """Zuordnung 3 x 3, Massen 1, Kosten Zeilen (4, 1, 3), (2, 0, 5), (3, 2, 2). Handrechnung: die sechs Zuordnungen kosten 5 / 6 / 6 / 7 / 9 / 11,
    die beste (Lager 1 -> Kunde 2, Lager 2 -> Kunde 1, Lager 3 -> Kunde 3) kostet 1 + 2 + 2 = 5."""
    return handbeispiel_zuordnung()


@pytest.fixture
def hand2():
    """Transport 3 x 3, Angebote (9, 6, 5), Nachfragen (8, 8, 4), Kosten Zeilen (4, 20, 29), (20, 4, 29), (5, 5, 4); Optimum 97 mit Plan Zeilen (8, 1, 0), (0, 6, 0), (0, 1, 4)."""
    return handbeispiel_transport()
