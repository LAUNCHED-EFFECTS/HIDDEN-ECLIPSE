"""Renders tracked positions onto an interactive orthographic globe."""

from __future__ import annotations

import plotly.graph_objects as go

from geo import Position, great_circle_km, great_circle_path, initial_bearing_deg, midpoint

# Identity is carried by shape and label as well as hue, so the picture still
# reads without color: hostile is a red diamond, friendly a blue circle.
HOSTILE_COLOR = "#e66767"
FRIENDLY_COLOR = "#3987e5"
ARC_COLOR = "#8a8a82"

OCEAN = "#12161c"
LAND = "#242a33"
COASTLINE = "#4a525e"
SURFACE = "#0d1117"
TEXT_PRIMARY = "#f0f2f5"
TEXT_SECONDARY = "#a6adb8"


def _marker_trace(pos: Position, name: str, color: str, symbol: str) -> go.Scattergeo:
    return go.Scattergeo(
        lat=[pos.lat],
        lon=[pos.lon],
        name=name,
        mode="markers+text",
        marker=dict(
            size=14,
            color=color,
            symbol=symbol,
            # 2px surface ring keeps the mark legible against land or ocean.
            line=dict(width=2, color=SURFACE),
        ),
        text=[name],
        textposition="top center",
        textfont=dict(size=12, color=TEXT_PRIMARY),
        hovertemplate=f"<b>{name}</b><br>%{{lat:.4f}}°, %{{lon:.4f}}°<extra></extra>",
    )


def build_globe(
    hostile: Position,
    friendly: Position,
    hostile_name: str = "RED",
    friendly_name: str = "BLUE",
) -> go.Figure:
    """Build the globe figure with both positions and the arc between them."""
    separation = great_circle_km(hostile, friendly)
    bearing = initial_bearing_deg(friendly, hostile)
    center = midpoint(hostile, friendly)
    arc = great_circle_path(friendly, hostile)

    fig = go.Figure()

    # Drawn first so the markers sit on top of it.
    fig.add_trace(
        go.Scattergeo(
            lat=[p.lat for p in arc],
            lon=[p.lon for p in arc],
            name=f"Great-circle track · {separation:,.0f} km",
            mode="lines",
            line=dict(width=2, color=ARC_COLOR, dash="dot"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(_marker_trace(hostile, hostile_name, HOSTILE_COLOR, "diamond"))
    fig.add_trace(_marker_trace(friendly, friendly_name, FRIENDLY_COLOR, "circle"))

    fig.update_geos(
        projection=dict(
            type="orthographic",
            # Aim the camera at the midpoint so both marks face the viewer
            # whenever they are on the same hemisphere.
            rotation=dict(lat=center.lat, lon=center.lon, roll=0),
        ),
        showland=True,
        landcolor=LAND,
        showocean=True,
        oceancolor=OCEAN,
        showcoastlines=True,
        coastlinecolor=COASTLINE,
        coastlinewidth=0.6,
        showcountries=True,
        countrycolor=COASTLINE,
        showlakes=False,
        showframe=False,
        lataxis=dict(showgrid=True, gridcolor="#1d232b", gridwidth=0.5, dtick=15),
        lonaxis=dict(showgrid=True, gridcolor="#1d232b", gridwidth=0.5, dtick=15),
        bgcolor=SURFACE,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Tactical picture<br>"
                f"<span style='font-size:13px;color:{TEXT_SECONDARY}'>"
                f"{friendly_name} → {hostile_name}: {separation:,.0f} km at {bearing:.0f}° true"
                "</span>"
            ),
            font=dict(size=20, color=TEXT_PRIMARY),
            x=0.03,
            y=0.95,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.04,
            xanchor="left",
            x=0.02,
            font=dict(color=TEXT_SECONDARY, size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor=SURFACE,
        margin=dict(l=0, r=0, t=90, b=40),
        height=760,
        dragmode="orbit",
    )
    return fig
