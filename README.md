# Vom Zuordnen zum Transportieren – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-zuordnen-transportieren-demo.streamlit.app/)**

Teil der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es ist die Wurzel der Reihe: Es setzt nichts anderes voraus, und die weiteren Demos der Reihe setzen an den Annahmen an, die hier stehen bleiben (Abschnitt „Ehrliche Grenzen“). Vehikel ist der Ort: **Lager** mit Angeboten, **Kunden** mit Nachfragen, Kosten je Einheit; gesucht ist der billigste **Transportplan**. Der Löser ist ein eigener Transportsimplex in numpy (`ztr_simplex.py`).

## Warum dieses Problem

Eine Zuordnung („jedes Lager liefert an genau einen Kunden“) und ein Transportplan („Lager liefern Mengen an mehrere Kunden“) sind bei gleichen Massen und gleich vielen Lagern und Kunden dieselbe Aufgabe: der Optimalwert ist derselbe, und es gibt immer eine optimale Zuordnung als Ecke. Die Demo zeigt, **wann** die Antwort eine Zuordnung ist, **wie viele Lager der Plan spaltet** und **wie viel Masse** das betrifft (die Zählung allein täuscht: schon bei Massen, die nur mit dem Gewicht $s = 0{,}001$ zufällig gemischt sind, sind im Mittel etwa drei Viertel der Lager gespalten, aber nur etwa 0,14 % der Masse liegen außerhalb des Hauptabnehmers; Punkte im Einheitsquadrat, $n = 20$, die einzelnen Massen weichen dabei typisch um etwa 0,1 % vom Mittel ab), und dass der gezeigte optimale Plan bei Gleichstand **eine Auswahl** unter mehreren gleich guten ist. „Lager spalten“ ist hier der Hausbegriff für eine Zeile des Plans mit mehr als einem Eintrag; ein Plan ohne gespaltenes Lager ist eine **Monge-Abbildung** (englisch Monge map).

## Modell

Gegeben sind $n$ Lager mit Angeboten $a_i$, $m$ Kunden mit Nachfragen $b_j$ (Summe der Angebote gleich Summe der Nachfragen) und Kosten $c_{ij}$ je Einheit. Gesucht ist ein Plan $x_{ij} \ge 0$ mit Zeilensummen $a_i$ und Spaltensummen $b_j$ (**Randverteilungen**, englisch marginals), der $\sum c_{ij} x_{ij}$ minimiert. Das Duale (Preise $u_i$, $v_j$ mit $u_i + v_j \le c_{ij}$) steht im 📐-Expander der App, dort mit einer Lesehilfe und einer Symboltabelle. Eine ausführliche, verständlich geschriebene Erklärung mit Fachteil steht auf der Seite [Vom Zuordnen zum Transportieren verständlich erklärt](https://sebastianhanisch.net/transportplan-erklaert.html); die App fasst sie im Expander „💬 In einfachen Worten“ zusammen.

- **Instanzen:** Gesamtmenge 1 auf beiden Seiten. Massen $a = (1 - s)/n + s \cdot \text{Dirichlet}(1)$, ebenso $b$ (Dirichlet(1) = unabhängige Exponentialzahlen, durch ihre Summe geteilt); $s = 0$ gibt gleiche Massen $1/n$, $s = 1$ unabhängig gezogene Dirichlet(1)-Massen.
- **Kostenfamilien:** Punkte im Einheitsquadrat (quadrierter Abstand, geteilt durch den größten Wert), stetige Zufallskosten $U(0,1)$, ganzzahlig 0 bis 100 / 0 bis 10 / 0 bis 2, Gitter L1 (Summe der Beträge), Gitter L2² (quadrierter euklidischer Abstand). Gitter: Seitenlänge Standard aufgerundete Wurzel aus $n + m$, plus 1 (6 bei $n = m = 10$); Ziehung ohne Zurücklegen (alle Orte verschieden) oder mit Zurücklegen (auch doppelte Orte); die Gitterseite ist ein Regler, weil der Anteil der Instanzen mit mehreren optimalen Plänen stark an ihr hängt.
- **Zählweise:** „gespaltenes Lager“ = Zeile des Plans mit mehr als einem Eintrag über $10^{-12}$ (Gesamtmenge 1); „Nebenmasse“ = Anteil der Gesamtmasse, den ein Lager nicht an seinen größten Abnehmer liefert. **Bei jeder Aussage über gespaltene Lager stehen Zählung und Masse nebeneinander.** Zähleinheit jeder Messung ist die **Instanz**.

## Methodik

- **Kern** (`ztr_simplex.py`, nur numpy): Startecke nach der Nordwest-Ecken-Regel, Preise aus dem Basisbaum, reduzierte Kosten, ein Eintritt nach der gewählten **Eintrittsregel** (erste, letzte oder zufällige Zelle mit negativen reduzierten Kosten; die Reihen rechnen mit „steilste“, der größten Abnahme), der Kreis im Baum bestimmt die austretende Zelle. Bei Entartung (Zeile und Spalte werden gleichzeitig leer) bleibt die Basis ein Baum mit Zellen der Menge 0; die Regel „erste“ mit kleinstem Index beim Austritt ist die Regel von Bland, die nach dem Lehrbuchsatz (in exakter Arithmetik) nicht kreist; die Zelltoleranzen sind Gleitkomma, ein Zyklus ist deshalb nur empirisch ausgeschlossen (in den Tests mit stark entarteten Zuordnungen ist kein Lauf in einen Zyklus geraten); die Regeln „zufällige“ und „steilste“ wechseln nach 25 entarteten Schritten in Folge zu ihr.
- **Ungarische Methode** (`ztr_zuordnung.py`, Kuhn-Munkres mit kürzesten erweiternden Wegen): Vergleich bei gleichen Massen; ein von der Simplex-Rechnung unabhängiger Rechenweg.
- **Eindeutigkeitstest** (`ztr_eindeutigkeit.py`, ohne Linearoptimierungs-Bibliothek): Die optimale Menge besteht aus allen zulässigen Plänen auf den Zellen mit reduzierten Kosten 0; der Plan ist genau dann eindeutig, wenn es auf diesen Zellen keinen Kreis von Verschiebungen gibt, der nur Zellen mit Menge größer null verkleinert (Erreichbarkeit im Pfeilgraphen, starke Zusammenhangskomponenten). Exakt ist der Test bei ganzzahligen und Gitterkosten; bei stetigen Kosten gilt eine Zelle mit reduzierten Kosten bis zur Toleranz $10^{-9}$ (relativ) als Nullzelle.
- **Zufall:** Hausgenerator SplitMix64 (`ztr_rng.py`, auf Windows und Linux bitgleich); je Zweck und Index ein eigener Zufallsstrom, je Massenungleichheit $s$ eigene Instanzen.
- **Messreihen:** `tools/messreihen.py` rechnet die Reihen dieser README mit dem Code der Demo neu und schreibt `tools/messreihen.json` (Zufalls-Seed 2026, Eintrittsregel „steilste“), `tools/messreihen_fremdloeser.py` die Reihe der Fremdlöser; `tests/test_claims.py` prüft jede Zahl dieser Reihen gegen die Dateien und rechnet die kleinen Reihen neu.

## Befunde (gemessen, keine Behauptungen)

Die Zahlen der Reihen stammen aus `tools/messreihen.json` und `tools/messreihen_fremdloeser.json`, die Handbeispiele aus Handrechnung und Tests; „Instanz“ ist immer die Zähleinheit. Die Zählungen gelten für **ausgeglichene Massen** (Summe Angebot gleich Summe Nachfrage), **Ecken des Simplex**, diese Kostenfamilien, diese Zufallszüge für die Massen und diese Reihen; mathematisch ist das Lehrbuchwissen (Birkhoff, Kantorovich); die Demo zählt und zeigt es an Instanzen.

**1. Gleiche Massen 1/n: die Ecke ist eine Zuordnung.** In **3 640 von 3 640 Instanzen** (sieben Kostenfamilien, $n$ = 5 / 10 / 20 / 40 mit 200 / 200 / 100 / 20 Instanzen je Familie und $n$) ist die Ecke eine Permutationsmatrix; der Wert gleicht dem der Ungarischen Methode in 3 640 von 3 640 Instanzen (größte Abweichung 5,7e-14). Das gilt für Simplex-Ecken; ein Innere-Punkte-Verfahren ohne Crossover (Ergebnis in der Mitte der optimalen Fläche) ist nicht gemessen.

**2. Lager spalten: Zählung gegen Masse.** Punkte im Einheitsquadrat, $n = m = 20$, Massen $a = (1 - s)/n + s \cdot \text{Dirichlet}(1)$, ebenso $b$ (unabhängig gezogen), je $s$ eigene 100 Instanzen, Eintrittsregel „steilste“:

| s | gespaltene Lager von 20: Mittel [Min, Max] | Nebenmasse (Mittel) | Monge-Abbildung | Einträge = n + m − 1 = 39 |
|---|---|---|---|---|
| 0 | 0,00 [0, 0] | 0,000 % | 100 von 100 | 0 von 100 |
| 0,001 | 15,20 [12, 19] | 0,142 % | 0 von 100 | 100 von 100 |
| 0,01 | 14,89 [12, 19] | 1,32 % | 0 von 100 | 100 von 100 |
| 0,03 | 15,35 [11, 18] | 4,36 % | 0 von 100 | 100 von 100 |
| 0,1 | 14,95 [11, 18] | 12,91 % | 0 von 100 | 100 von 100 |
| 0,3 | 14,49 [10, 17] | 23,24 % | 0 von 100 | 100 von 100 |
| 1 | 10,02 [6, 14] | 28,76 % | 0 von 100 | 100 von 100 |

Die **Zählung springt** beim Schritt von $s = 0$ zu $s = 0{,}001$ (von 0 auf etwa 15 von 20 Lagern), die **Masse wächst stetig**: Nebenmasse geteilt durch $s$ ist 1,42 / 1,32 / 1,45 / 1,29 für $s$ = 0,001 / 0,01 / 0,03 / 0,1, also etwa 1,3 bis 1,45 (die Streuung ist überwiegend Stichprobenrauschen: andere Zufalls-Seeds streuen das Mittel über 100 Instanzen um etwa ±0,05; mit 1 000 Instanzen ergeben sich unten 1,39 und 1,37), danach nicht mehr proportional. Bei $s = 0$ hat der Plan nur $n = 20$ Einträge (entartet, eine Permutation), bei $s > 0$ in allen 100 Instanzen $n + m - 1 = 39$. Mit **1 000** Instanzen (anderer Zufalls-Seed, 2027): $s = 0{,}001$ gibt 15,21 gespaltene Lager und 0,139 % Nebenmasse, $s = 0{,}03$ gibt 15,11 und 4,110 %; in keiner dieser 2 000 Instanzen liefert jedes Lager an genau einen Kunden. Bei stetigen Zufallskosten ($U(0,1)$), $s = 1$, 200 Instanzen: 10,08 gespaltene Lager [5, 15], 29,511 % Nebenmasse, Monge-Abbildung in 0 von 200. Abhängigkeit von $n$ bei $s = 1$ (Punkte im Einheitsquadrat, je 60 Instanzen): 2,48 / 4,93 / 9,63 / 20,18 gespaltene Lager für $n$ = 5 / 10 / 20 / 40 (also etwa die Hälfte) und 26,6 / 29,0 / 30,5 / 29,7 % Nebenmasse.

**„Bei ungleichen Massen nie eine Zuordnung“ gilt nicht allgemein.** Gemessen ist: bei diesen zufällig gezogenen, verschiedenen Massen liefert in 0 von 100 Instanzen je $s > 0$ jedes Lager an genau einen Kunden. Sind die Massen auf beiden Seiten gleich verteilt oder passend vertauscht, kann das Optimum eine Zuordnung ohne Spaltung sein: Angebote und Nachfragen $(8, 8, 4)$ mit den Kosten des Handbeispiels geben einen Diagonalplan (Kosten 80, drei Einträge), und Angebote $(1, 2)$ gegen Nachfragen $(2, 1)$ mit Kosten $\begin{pmatrix}1&0\\0&1\end{pmatrix}$ haben den eindeutigen Plan Lager 1 → Kunde 2, Lager 2 → Kunde 1 (Kosten 0). Beides prüfen `tests/test_simplex.py` und die Aufzählung aller Ecken.

**3. Welcher Plan? Gleichstände** (Zeilen mit $s = 0$: Zuordnung mit gleichen Massen; Zeilen mit $s = 1$: Transport mit unabhängig gezogenen Dirichlet(1)-Massen; „–“ = keine Instanz mit verschiedenen Plänen, das Mittel ist dann nicht definiert; „mehrere“ = der exakte Eindeutigkeitstest findet mehr als einen optimalen Plan; „verschieden“ = die drei Eintrittsregeln „erste“, „letzte“, „zufällige“ liefern bei gleichem Wert nicht denselben Plan; „zweite Kennzahl“ = Summe aus einer zweiten Kostenmatrix mal Plan, relative Spanne über die drei Pläne, Mittel über die Instanzen mit verschiedenen Plänen):

| Kostenfamilie | n | s | Gitterseite | Ziehung | Instanzen | mehrere optimale Pläne | Regeln wählen verschiedene Pläne | zweite Kennzahl streut um |
|---|---|---|---|---|---|---|---|---|
| stetig (Zufallskosten) | 10 | 0 | – | – | 300 | 0 von 300 | 0 von 300 | – |
| ganzzahlig 0 bis 10 | 10 | 0 | – | – | 300 | 168 von 300 | 137 von 300 | 13,2 % |
| ganzzahlig 0 bis 2 | 10 | 0 | – | – | 300 | 298 von 300 | 294 von 300 | 23,7 % |
| Gitter L1 | 10 | 0 | 6 | ohne Zurücklegen | 300 | 299 von 300 | 290 von 300 | 13,7 % |
| Gitter L2² | 10 | 0 | 6 | ohne Zurücklegen | 300 | 223 von 300 | 188 von 300 | 1,6 % |
| Gitter L1 | 10 | 0 | 6 | mit Zurücklegen | 150 | 150 von 150 | 150 von 150 | 12,2 % |
| Gitter L1 | 10 | 0 | 10 | mit Zurücklegen | 150 | 148 von 150 | 142 von 150 | 10,0 % |
| Gitter L1 | 10 | 0 | 20 | mit Zurücklegen | 150 | 144 von 150 | 134 von 150 | 7,6 % |
| Gitter L2² | 10 | 0 | 6 | mit Zurücklegen | 150 | 150 von 150 | 140 von 150 | 1,5 % |
| Gitter L2² | 10 | 0 | 10 | mit Zurücklegen | 150 | 125 von 150 | 109 von 150 | 0,8 % |
| Gitter L2² | 10 | 0 | 20 | mit Zurücklegen | 150 | 53 von 150 | 43 von 150 | 0,9 % |
| stetig (Zufallskosten) | 15 | 1 | – | – | 150 | 0 von 150 | 0 von 150 | – |
| Gitter L1 | 15 | 1 | 7 | ohne Zurücklegen | 150 | 150 von 150 | 150 von 150 | 6,7 % |

Die Zahl der Instanzen mit mehreren optimalen Plänen **hängt an Familie, Gitterseite und Ziehung der Punkte** (Gitter L2² mit Zurücklegen: 150 von 150 bei Seite 6, 53 von 150 bei Seite 20); jeder Satz darüber nennt deshalb Familie, Gitterseite, Ziehung, $n$ und die Zahl der Instanzen. **Es gibt keinen Fehler, nur eine Auswahl:** In allen gemessenen Instanzen aller Zeilen ist der Wert der drei Pläne derselbe (größte Abweichung unter $10^{-13}$), der Plan kann verschieden sein (bei den Zeilen mit Gleichstand meist, nicht immer), und eine zweite Kennzahl kann zwischen den gleich teuren Plänen verschieden sein. Bei stetigen Kosten ist der Plan in den gemessenen Instanzen eindeutig; die Frage, welche Ecke gewählt wird, entsteht erst bei Gitter- und ganzzahligen Kosten. Die Zahlen gelten für diese drei Eintrittsregeln dieses Simplex; andere Löser und andere Versionen können bei Gleichstand andere Ecken wählen (nicht gemessen).

**3a. Verschiedene Löser, verschiedene Pläne.** Fünf Fremdlöser auf denselben Zuordnungsinstanzen ($n = 10$, gleiche Massen, 150 Instanzen je Familie): `scipy.optimize.linprog` mit HiGHS-Simplex und mit HiGHS-Innere-Punkte (mit Crossover), `scipy.optimize.linear_sum_assignment`, der Mindestkostenfluss von OR-Tools und `networkx.network_simplex` (numpy 2.5.3, scipy 1.18.1, networkx 3.7, OR-Tools 9.15; `tools/messreihen_fremdloeser.py`, Datei `tools/messreihen_fremdloeser.json`). In jeder dieser 900 Instanzen ist der Optimalwert der fünf Löser derselbe (bis auf 1e-6 relativ); mindestens zwei der fünf Pläne sind verschieden in 0 / 14 / 86 / 148 / 148 / 113 von 150 Instanzen bei stetigen Kosten / ganzzahlig 0 bis 100 / ganzzahlig 0 bis 10 / ganzzahlig 0 bis 2 / Gitter L1 / Gitter L2² (Gitterseite 6, ohne Zurücklegen). Die Fremdlöser gehören nicht zur Laufzeit der Demo; die Zahl gilt für diese Löser in diesen Versionen, und welche Ecke ein Löser bei Gleichstand wählt, kann sich zwischen Versionen und Plattformen ändern.

**4. Preise (Dualität).** Über 100 Zufallsinstanzen (Punkte im Einheitsquadrat, $s = 1$, je 25 Instanzen mit $n = m$ = 3 / 8 / 20 / 40): größte Abweichung |Primalwert − Dualwert| 4,4e-16 und größte Verletzung von $u_i + v_j \le c_{ij}$ 5,6e-17 (Rundung; das ist eine Selbstprobe mit den Preisen des eigenen Simplex, die unabhängige Gegenprobe steht in den Tests). Handbeispiel (Angebote 9, 6, 5; Nachfragen 8, 8, 4; quadrierte Abstände): Optimum 97, Preise $u = (0, -16, -15)$ und $v = (4, 20, 19)$ mit $u_1 = 0$, Schlupf $c_{ij} - u_i - v_j$ null genau auf den fünf Zellen des Plans, Dualwert 97. Eine Einheit mehr bei Lager 3 und Kunde 1 ändert die Kosten um $u_3 + v_1 = -11$ (97 → 86, durch erneutes Lösen und durch Aufzählen aller 16 Ecken der geänderten Aufgabe bestätigt). Im Handbeispiel (nicht entartet) sind die Preise nur bis auf eine Konstante bestimmt (die fünf Gleichungen auf den Zellen des Plans haben Rang 5 bei sechs Unbekannten); bei Entartung, etwa bei gleichen Massen, gibt es weitere optimale Preise: im Handbeispiel Zuordnung $3 \times 3$ liegt bei $u_1 = 0$ das optimale $u_2$ im Bereich $[-2, -1]$ und das optimale $u_3$ im Bereich $[-1, 0]$ (Dual-Linearprogramm, im Test nachgerechnet).

**5. Handbeispiele von Hand gerechnet.** Zuordnung $3 \times 3$ (Kosten Zeilen (4, 1, 3), (2, 0, 5), (3, 2, 2)): die sechs Zuordnungen kosten 5 / 6 / 6 / 7 / 9 / 11, die beste ist eindeutig und kostet 5. Transport (9, 6, 5) gegen (8, 8, 4): Optimum 97 mit zwei von drei gespaltenen Lagern, Nebenmasse 2 von 20 = 10 %, 5 Einträge, keine der 27 Abbildungen Lager → Kunde hat das Bildmaß (8, 8, 4); das Transportpolytop hat 18 Ecken. Gleichstand (Gitterpunkte, L1): zwei beste Zuordnungen mit Kosten 3, jede Mischung $t P_1 + (1 - t) P_2$ kostet ebenfalls 3, und eine zweite Größe (quadrierte waagerechte Verschiebung) ist 0, 2 und 1 für die beiden Ecken und die Mitte.

### Zahl → Test

| Zahl der README | Test in `tests/test_claims.py` |
|---|---|
| 3 640 von 3 640, 5,7e-14 | `test_zuordnen_gesamtzahl_und_wertabweichung`, `test_zuordnen_n5_neu_gerechnet` (Band) |
| Tabelle zu 2., Nebenmasse geteilt durch $s$ | `test_uebergang_tabelle_der_readme`, `test_uebergang_neu_gerechnet` (Band), `test_nebenmasse_durch_s` |
| 1 000 Instanzen, stetige Kosten, $n$-Abhängigkeit | `test_uebergang_zusatzreihen` |
| Tabelle zu 3. | `test_gleichstand_tabelle_der_readme`, `test_gleichstand_neu_gerechnet` (Band) |
| 3a. fünf Fremdlöser | `test_fremdloeser_reihen_der_readme`, `test_fremdloeser_neu_gerechnet` (Band, nur mit SciPy, NetworkX und OR-Tools) |
| 100 Instanzen Dualität | `test_dualitaet_zeile` |
| Handwerte (97, 80, 5, 10 %, −11, 86, 3) | `test_handwerte_der_readme` (dazu `tests/test_simplex.py`, Aufzählung in Brüchen) |
| Zahlen der Preset-Hilfen | `test_preset_hilfetexte_nennen_gemessene_zahlen` |
| Testzahl | `test_testzahl_der_readme` |

Die Reihen mit vielen Instanzen (1 000 Instanzen, $n = 40$) stehen in der Datei und werden nicht jedes Mal neu gerechnet; die kleinen Reihen rechnet der Test neu und vergleicht als Band (Zählungen ±1, Mittel ±0,05), weil die Wahl der Ecke bei Gleichstand auf anderen Plattformen anders ausfallen kann.

## Ehrliche Grenzen

- **Gültigkeit:** Die Zählungen gelten für ausgeglichene Massen, Simplex-Ecken, die genannten Kostenfamilien, diese Zufallszüge und die Zähleinheit Instanz. Es wird **nicht** behauptet, dass ungleiche Massen immer Lager spalten (Gegenbeispiele oben).
- **Annahmen, die hier stehen bleiben:** (1) Der optimale Plan ist eine dünne Ecke (höchstens $n + m - 1$ Einträge), und bei Gleichstand wählt der Löser eine. (2) Gesucht ist das Minimum bei festen Randsummen (nicht die Projektion einer alten Matrix auf neue Randsummen). (3) Lager, Kunden und Kosten sind gegeben; der Optimalwert ist das Ende der Rechnung, ein Preis für einen Auftrag, kein Abstand zwischen Verteilungen. (4) Angebot und Nachfrage sind in der Summe gleich; der Löser lehnt ungleiche Summen mit einer Fehlermeldung ab (`ValueError`), die App bietet sie nicht an. (5) Bei Gleichstand ist die Wahl des optimalen Plans für den Wert gleichgültig; für ein zweites Kriterium ist sie es nicht (Experiment 3), und eine Zwischenverteilung kann von ihr abhängen (hier nicht gerechnet). Weitere Demos der Reihe gehen auf diese Annahmen ein.
- **Nicht gemessen:** Innere Punkte ohne Crossover (Mitte der Fläche statt Ecke); andere Löser und Versionen bei Gleichstand; Rechenaufwand (kein Vergleich nach Schritten oder Zeit mit anderen Verfahren); Kostenfamilien und Zufallszüge jenseits der genannten.
- **Literatur:** Der Satz von Birkhoff und die Dualität des Transportproblems stehen in Lehrbüchern zur Linearen Optimierung; die Demo belegt sie nur an Beispielen und durch den Vergleich mit unabhängigen Verfahren in den Tests. Originalquellen wurden nicht nachgelesen (Quelle nicht eingesehen, deshalb kein Literaturtest).

## Verifikation

- **Kern:** Handbeispiele von Hand gerechnet (Zuordnung, Transport mit zwei gespaltenen Lagern, Gleichstand mit zwei besten Zuordnungen, Preise und Preisprobe), Aufzählung aller Ecken des Transportpolytops in Brüchen über alle aufspannenden Bäume (ohne Simplex), Randfälle ($n = 1$ oder $m = 1$, Lager mit Menge 0, alle Kosten gleich, Massen unter $10^{-7}$, ungleiche Summen, ungültige Eingaben), Bland-Rückfall bei stark entarteten Zuordnungen; die Einheiten (Startecke, Preise, Kreis, Pivot, Eintrittsregeln) an Mini-Instanzen.
- **Orakel** (`tests/test_oracle_transportsimplex.py`): Optimalwert gegen `scipy.optimize.linprog` (HiGHS, Simplex und Innere Punkte), `networkx.network_simplex` und den Mindestkostenfluss von OR-Tools (ganzzahlige Instanzen mit Abweichung 0), bei gleichen Massen gegen `linear_sum_assignment` und die eigene Ungarische Methode; starke Dualität gegen ein unabhängig gelöstes Dual-LP und die c-Transformierte; Eindeutigkeitstest gegen die Aufzählung aller 120 Permutationen ($n = 5$, Kosten 0 bis 2, 300 Instanzen) und gegen ein zweites Linearprogramm auf der optimalen Fläche. Bei Gleichstand wird nie bitgleich verglichen, sondern geprüft, dass der Plan zur optimalen Fläche gehört.
- **Eigenschaften** (`tests/test_eigenschaften_transport.py`): Kosten mal Faktor oder plus Konstante, Umnummerieren, Massen mal Faktor, jede Ecke bei Massen $1/n$ eine Permutation (alle Familien, alle Regeln), $\sum_i (k_i - 1) = m - 1$, Mischungen optimaler Pläne, Verschiebung der Preise, Monotonie, gleicher Zufalls-Seed in einem zweiten Prozess (`PYTHONHASHSEED`).
- **Verdikte und Zahlen:** Verdikte sind reine Funktionen und an handgesetzten Zeilen geprüft (`tests/test_evaluation.py`); jede Zahl der Messreihen und Handbeispiele steht in `tests/test_claims.py` beziehungsweise `tests/test_simplex.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum, jede Option der Auswahlregler, Gitterseiten und Ziehungen, jedes Preset mit seiner erwarteten Verdikt-Zeile, Permalink), Layout- und Footer-Test (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf), Laufzeit-Importe gegen `requirements.txt`. Im Browser ist die Seite stichprobenartig durchgespielt (Presets, Scrollen); ein vollständiger Klick-Durchlauf steht aus.
- **Testzahl:** 207 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest und die Bibliotheken der Gegenproben (scipy, networkx, OR-Tools) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, plotly |
| `tools/messreihen.py` | rechnet die Messreihen der README neu und schreibt `tools/messreihen.json` |
| `tools/messreihen_fremdloeser.py` | fünf Fremdlöser auf denselben Zuordnungsinstanzen, schreibt `tools/messreihen_fremdloeser.json` |
| `ztr_constants.py` | Regler-Grenzen, Kostenfamilien, Farben, Presets und Hilfetexte |
| `ztr_eindeutigkeit.py` | exakter Eindeutigkeitstest für den optimalen Plan |
| `ztr_evaluation.py` | Kennzahlen, Reihen über feste Instanzen, Verdikte, Preise |
| `ztr_layout.py` | Fester Kopf (Stylesheet) |
| `ztr_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `ztr_rng.py` | Hausgenerator SplitMix64 (plattformgleich) |
| `ztr_scenario.py` | Instanzen: Massen, Kostenfamilien, Gitter, Handbeispiele |
| `ztr_simplex.py` | Transportsimplex mit Basisbaum (wiederverwendbar, ohne Streamlit) |
| `ztr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ztr_zuordnung.py` | Ungarische Methode (Kuhn-Munkres) |
| `tests/` | Handbeispiele, Aufzählung der Ecken in Brüchen, Orakel, Eigenschaften, Verdikte, Claims, Presets, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Kein Rechenaufwandsvergleich** mit anderen Verfahren (nach Schritten oder Zeit); die Demo zählt Pläne, nicht Rechenschritte.
- **Kein Innere-Punkte-Verfahren und keine Bibliothek zur Laufzeit** (`scipy`, `networkx` und OR-Tools nur in den Tests); der Löser ist der eigene Simplex.
- **Keine ungleichen Summen** von Angebot und Nachfrage (Teilbedienung, Strafen, Ausreißer); der Löser lehnt sie mit einer Fehlermeldung ab.
- **Kein PDF-Export.**
- **Keine Netzwerksimplex-Analyse** (Pivotzahlen, Basiswechsel im Graphen): dafür gibt es im Portfolio eine eigene Demo zum Netzwerksimplex; hier zählen nur Plan und Eckenwahl.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy; die Gegenproben der Tests nutzen SciPy, NetworkX und OR-Tools.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html).
