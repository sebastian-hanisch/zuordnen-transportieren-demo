"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Beschriftungen von Zellen und Pfeilen sind Annotationen (kein Text-Trace, keine texttemplate)."""

import numpy as np
import plotly.graph_objects as go

import ztr_constants as C
from ztr_simplex import TOL_MENGE

MAX_ZELLEN_MIT_ZAHL = 144      # bis 12 x 12 steht die Zahl in der Zelle


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.15))
    return lock_axes(fig)


def zahl_text(x: float, gesamt: float) -> str:
    """Zellzahl: ganze Zahl bei ganzzahligen Mengen, sonst Anteil an der Gesamtmenge in Prozent mit Komma."""
    if abs(x - round(x)) < 1e-9 and gesamt > 1.5:
        return str(int(round(x)))
    return f"{100.0 * x / gesamt:.1f}".replace(".", ",") + " %"


def build_plan_matrix(plan, a, b, titel="", hoehe=380, beschriftung=True):
    """Der Plan als Matrix (Zeilen = Lager, Spalten = Kunden); die Randverteilungen stehen in den Achsenbeschriftungen. Zellen mit Menge 0 bleiben leer."""
    plan = np.asarray(plan, float)
    n, m = plan.shape
    gesamt = float(plan.sum())
    schwelle = TOL_MENGE * max(1.0, gesamt)
    z = np.where(plan > schwelle, plan, np.nan)
    xs = [f"K{j + 1}" for j in range(m)]
    ys = [f"L{i + 1}" for i in range(n)]
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale=[[0, "#e3efed"], [1, C.COLOR_LAGER]], showscale=False, xgap=1, ygap=1,
                               hovertemplate="%{y} → %{x}: %{z:.4g}<extra></extra>", zmin=0, zmax=float(np.nanmax(z)) if np.isfinite(z).any() else 1.0))
    if beschriftung and n * m <= MAX_ZELLEN_MIT_ZAHL:
        for i in range(n):
            for j in range(m):
                if plan[i, j] > schwelle:
                    hell = plan[i, j] > 0.55 * plan.max()
                    fig.add_annotation(x=xs[j], y=ys[i], text=zahl_text(plan[i, j], gesamt), showarrow=False,
                                       font=dict(size=11 if n <= 8 else 9, color="white" if hell else "#1b3b37"))
    fig.update_xaxes(side="top", title_text="Kunden (Nachfrage)" if beschriftung else None, tickmode="array", tickvals=xs,
                     ticktext=[f"K{j + 1}<br>{zahl_text(b[j], gesamt)}" if m <= 12 else f"K{j + 1}" for j in range(m)], tickfont=dict(size=10))
    fig.update_yaxes(autorange="reversed", title_text="Lager (Angebot)" if beschriftung else None, tickmode="array", tickvals=ys,
                     ticktext=[f"L{i + 1} · {zahl_text(a[i], gesamt)}" if n <= 12 else f"L{i + 1}" for i in range(n)], tickfont=dict(size=10))
    fig = _base(fig, height=hoehe, legend=False)
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)), margin=dict(l=10, r=10, t=95, b=10))     # Platz zwischen Titel und den Spaltenbeschriftungen oben
    return fig


def build_plan_karte(lager, kunden, plan, titel="", hoehe=380):
    """Die Orte (Lager = Quadrate, Kunden = Kreise) und der Plan als Pfeile; die Breite des Pfeils wächst mit der Menge."""
    plan = np.asarray(plan, float)
    n, m = plan.shape
    schwelle = TOL_MENGE * max(1.0, float(plan.sum()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lager[:, 0], y=lager[:, 1], mode="markers+text" if n <= 12 else "markers", name="Lager", text=[f"L{i + 1}" for i in range(n)],
                             textposition="top center", marker=dict(symbol="square", size=11, color=C.COLOR_LAGER), hovertemplate="Lager %{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=kunden[:, 0], y=kunden[:, 1], mode="markers+text" if m <= 12 else "markers", name="Kunden", text=[f"K{j + 1}" for j in range(m)],
                             textposition="bottom center", marker=dict(symbol="circle", size=11, color=C.COLOR_KUNDE), hovertemplate="Kunde %{text}<extra></extra>"))
    groesste = float(plan.max()) if plan.size else 1.0
    for i in range(n):
        for j in range(m):
            if plan[i, j] > schwelle:
                fig.add_annotation(x=kunden[j, 0], y=kunden[j, 1], ax=lager[i, 0], ay=lager[i, 1], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                   arrowhead=2, arrowsize=1.0, arrowwidth=0.8 + 4.2 * plan[i, j] / groesste, arrowcolor=C.COLOR_PFEIL, standoff=5)
    spanne_x = [min(lager[:, 0].min(), kunden[:, 0].min()), max(lager[:, 0].max(), kunden[:, 0].max())]
    spanne_y = [min(lager[:, 1].min(), kunden[:, 1].min()), max(lager[:, 1].max(), kunden[:, 1].max())]
    rand = 0.08 * max(spanne_x[1] - spanne_x[0], spanne_y[1] - spanne_y[0], 1e-9)
    # Rand über zwei unsichtbare Marker erzwingen (nicht über explizite Bereiche zusammen mit scaleanchor)
    fig.add_trace(go.Scatter(x=[spanne_x[0] - rand, spanne_x[1] + rand], y=[spanne_y[0] - rand, spanne_y[1] + rand], mode="markers", marker=dict(opacity=0),
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x")
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return _base(fig, height=hoehe)


def build_uebergang(zeilen, s_text):
    """Zwei Abbildungen: gespaltene Lager (Mittel, Balken von Minimum bis Maximum) und Nebenmasse (Mittel in %), je Massenungleichheit s. Kategorien als Text."""
    xs = [s_text(z["s"]) for z in zeilen]
    n = zeilen[0]["n"]
    mittel = [z["gespalten_mittel"] for z in zeilen]
    fig1 = go.Figure(go.Bar(x=xs, y=mittel, marker_color=C.COLOR_LINE, name="Mittel",
                            error_y=dict(type="data", symmetric=False, array=[z["gespalten_max"] - m for z, m in zip(zeilen, mittel)],
                                         arrayminus=[m - z["gespalten_min"] for z, m in zip(zeilen, mittel)], color="rgba(0,0,0,0.55)"),
                            hovertemplate="s = %{x}: %{y:.1f} von " + str(n) + "<extra></extra>"))
    fig1.update_xaxes(title_text="Massenungleichheit s", type="category")
    fig1.update_yaxes(title_text=f"gespaltene Lager (von {n})", range=[0, n * 1.05])
    fig2 = go.Figure(go.Bar(x=xs, y=[100.0 * z["neben_mittel"] for z in zeilen], marker_color=C.COLOR_KUNDE, name="Mittel",
                            hovertemplate="s = %{x}: %{y:.2f} %<extra></extra>"))
    fig2.update_xaxes(title_text="Massenungleichheit s", type="category")
    fig2.update_yaxes(title_text="Masse außerhalb des Hauptabnehmers (%)", range=[0, 100])
    return _base(fig1, height=320, legend=False), _base(fig2, height=320, legend=False)
