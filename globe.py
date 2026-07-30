"""Renders tracked positions onto an interactive orthographic globe."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from geo import (
    Position,
    elevation_angle_deg,
    great_circle_km,
    great_circle_path,
    initial_bearing_deg,
    midpoint,
    slant_range_km,
)

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
        customdata=[[pos.alt_m, pos.alt_ft]],
        hovertemplate=(
            f"<b>{name}</b><br>"
            "%{lat:.4f}°, %{lon:.4f}°<br>"
            "altitude %{customdata[0]:,.0f} m (%{customdata[1]:,.0f} ft)"
            "<extra></extra>"
        ),
    )


def build_globe(
    hostile: Position,
    friendly: Position,
    hostile_name: str = "RED",
    friendly_name: str = "BLUE",
) -> go.Figure:
    """Build the globe figure with both positions and the arc between them."""
    ground_range = great_circle_km(hostile, friendly)
    slant = slant_range_km(hostile, friendly)
    bearing = initial_bearing_deg(friendly, hostile)
    elevation = elevation_angle_deg(friendly, hostile)
    center = midpoint(hostile, friendly)
    arc = great_circle_path(friendly, hostile)

    fig = go.Figure()

    # Trace 0 — the ground track. Added first so the markers sit on top of it,
    # and hidden until a marker is hovered (see HOVER_SCRIPT). "legendonly"
    # rather than False so the legend still names it and carries the range.
    fig.add_trace(
        go.Scattergeo(
            lat=[p.lat for p in arc],
            lon=[p.lon for p in arc],
            name=f"Ground track · {ground_range:,.0f} km",
            mode="lines",
            line=dict(width=2, color=ARC_COLOR, dash="dot"),
            hoverinfo="skip",
            visible="legendonly",
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
            # Plotly does not wrap title text — a single long line runs past
            # the paper's right edge and is clipped — so the readout is broken
            # across explicit lines rather than left to fit.
            text=(
                "Tactical picture<br>"
                f"<span style='font-size:13px;color:{TEXT_SECONDARY}'>"
                f"{friendly_name} → {hostile_name}<br>"
                f"{slant:,.0f} km slant · {ground_range:,.0f} km ground<br>"
                f"bearing {bearing:03.0f}° true · elevation {elevation:+.2f}°"
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
        # Non-zero left/right so nothing sits flush against the window edge,
        # and enough top for the four-line title block.
        margin=dict(l=16, r=16, t=120, b=40),
        # No explicit height: plotly pins the div to layout.height when it is
        # set, which caps the globe regardless of the window. Leaving it unset
        # lets write_globe size the div against the viewport instead.
        autosize=True,
        # Must be "pan". Plotly only attaches the geo subplot's drag and wheel
        # handlers on this exact value — "orbit" is a 3D-scene mode and leaves
        # the globe inert. On a clipped projection like orthographic, "pan"
        # rotates the sphere rather than sliding a flat map.
        dragmode="pan",
    )
    return fig


# Index of the ground-track trace, which HOVER_SCRIPT toggles. It is added
# first in build_globe, so this stays 0.
ARC_TRACE_INDEX = 0

# Hover-driven visibility is not expressible in plotly's declarative API, so it
# rides along in the exported HTML as a post-plot hook.
HOVER_SCRIPT = """
(function () {
    var gd = document.getElementById('{plot_id}');
    var ARC = %d;
    var shown = false;
    var pending = null;

    function set(state) {
        if (pending) { clearTimeout(pending); pending = null; }
        if (shown === state) { return; }   // never restyle a no-op: each one
        shown = state;                     // redraws the globe
        Plotly.restyle(gd, {visible: state ? true : 'legendonly'}, [ARC]);
    }

    gd.on('plotly_hover', function () { set(true); });

    // Deferred, so sliding the cursor straight from one marker to the other
    // does not blink the track off between them. The redraw from set(true)
    // can itself emit an unhover, which this delay also absorbs.
    gd.on('plotly_unhover', function () {
        pending = setTimeout(function () { pending = null; set(false); }, 120);
    });
})();
""" % ARC_TRACE_INDEX


# scroll_zoom is on by default for geo, but it is stated here so that adding
# any other config key later cannot silently drop it.
GLOBE_CONFIG = {
    "scrollZoom": True,
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


# Sizes the plot to the viewport so the globe grows with the window.
#
# The !important is load-bearing: plotly writes `style="height:100%; width:100%"`
# onto the div itself, and an inline style beats a plain rule. (Its own
# default_width/default_height arguments are accepted but ignored, so CSS is the
# only lever.) The margin reset matters too — 100vh plus the browser's default
# 8px body margin overflows the viewport and raises a scrollbar.
PAGE_STYLE = f"""
<style>
  html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
    overflow: hidden;
    background: {SURFACE};
  }}
  /* Percentages, not vw/vh: 100vw counts the vertical scrollbar's width as
     part of the viewport, so it renders wider than the space actually
     available and the right edge gets clipped. 100% resolves against the
     body box, which excludes it. */
  .plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
  }}
</style>
"""


# Fraction of the plot area's width the globe should span. 1.0 fills it edge to
# edge; a circle cannot widen without growing taller, so filling the width crops
# the poles. Drop toward ~0.8 to keep more of the sphere in frame.
GLOBE_WIDTH_FILL = 1.0

# At projection.scale 1 plotly fits the sphere to the *shorter* side of the plot
# area — its projection fit takes Math.min of the width and height ratios — so a
# wide window leaves gaps either side. Correcting that needs the window's aspect
# ratio, which is only known in the browser.
FIT_SCRIPT = """
(function () {
    var gd = document.getElementById('{plot_id}');
    var FILL = %r;
    var applied = null;

    function fit() {
        var size = gd._fullLayout && gd._fullLayout._size;
        if (!size || !size.w || !size.h) { return; }

        var scale = Math.max(1, FILL * size.w / size.h);
        // Never below 1: that would shrink the globe rather than fill.
        if (applied !== null && Math.abs(scale - applied) < 0.01) { return; }
        applied = scale;

        Plotly.relayout(gd, {'geo.projection.scale': scale}).then(function () {
            // Plotly records the double-click reset target on first render
            // only, so without this the reset snaps back to the unfilled
            // globe. Guarded: if the internal shape changes this is a no-op,
            // not a crash.
            var sp = gd._fullLayout.geo && gd._fullLayout.geo._subplot;
            if (sp && sp.viewInitial) { sp.viewInitial['projection.scale'] = scale; }
        });
    }

    fit();
    // Debounced, and a no-op when the aspect ratio is unchanged — which also
    // means a resize that keeps the shape leaves a user's own zoom alone.
    window.addEventListener('resize', function () { setTimeout(fit, 150); });
})();
""" % GLOBE_WIDTH_FILL


def write_globe(fig: go.Figure, path: Path) -> None:
    """Write the figure to a standalone, full-window HTML page."""
    html = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        post_script=[HOVER_SCRIPT, FIT_SCRIPT],
        config=GLOBE_CONFIG,
    )
    html = html.replace("</head>", f"{PAGE_STYLE}</head>", 1)
    Path(path).write_text(html, encoding="utf-8")
