"""Renders tracked positions onto an interactive orthographic globe."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from defenses import DefenseSite
from geo import (
    Position,
    circle_path,
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

# Defence rings share the hostile hue but sit well back from it in weight, so
# the target marker still reads as the brightest thing on the map.
THREAT_LINE = "#b8524f"
THREAT_FILL = "rgba(230, 103, 103, 0.10)"

# Green on success, amber on failure — so the outcome is legible from the map
# alone, not only from the title. Green also keeps the route distinct from
# BLUE's own marker, which the old blue route sat right on top of.
ROUTE_COLOR = "#3ecf75"
ROUTE_FAILED_COLOR = "#eda100"

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


def _threat_ring(site: DefenseSite) -> go.Scattergeo:
    """A filled circle at the site's engagement radius."""
    ring = circle_path(site.position, site.kind.engagement_km)
    return go.Scattergeo(
        lat=[p.lat for p in ring],
        lon=[p.lon for p in ring],
        mode="lines",
        line=dict(width=1, color=THREAT_LINE),
        fill="toself",
        fillcolor=THREAT_FILL,
        legendgroup="defenses",
        showlegend=False,
        hoverinfo="skip",  # the site marker carries the detail
    )


def _defense_marker_trace(sites: list[DefenseSite]) -> go.Scattergeo:
    """One trace for every site, so the legend gets a single entry."""
    return go.Scattergeo(
        lat=[s.position.lat for s in sites],
        lon=[s.position.lon for s in sites],
        name=f"Air defence · {len(sites)} sites",
        mode="markers",
        marker=dict(
            size=9,
            color=THREAT_LINE,
            symbol="x-thin",
            line=dict(width=2, color=THREAT_LINE),
        ),
        legendgroup="defenses",
        customdata=[
            [s.designator, s.kind.name, s.kind.engagement_km, s.kind.ceiling_m]
            for s in sites
        ],
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
            "%{lat:.4f}°, %{lon:.4f}°<br>"
            "engagement %{customdata[2]:,.0f} km · ceiling %{customdata[3]:,.0f} m"
            "<extra></extra>"
        ),
    )


def _route_traces(plan) -> list[go.Scattergeo]:
    """The flown route plus its waypoints, as two traces."""
    track = plan.track
    ok = plan.succeeded
    return [
        go.Scattergeo(
            lat=[p.lat for p in track],
            lon=[p.lon for p in track],
            name=f"Planned route · {plan.route_km:,.0f} km",
            mode="lines",
            line=dict(width=2.5, color=ROUTE_COLOR if ok else ROUTE_FAILED_COLOR),
            legendgroup="route",
            hoverinfo="skip",
        ),
        go.Scattergeo(
            lat=[w.position.lat for w in plan.waypoints],
            lon=[w.position.lon for w in plan.waypoints],
            name=f"Waypoints · {len(plan.waypoints)}",
            mode="markers",
            marker=dict(
                size=6,
                color=ROUTE_COLOR if ok else ROUTE_FAILED_COLOR,
                symbol="circle",
                line=dict(width=1, color=SURFACE),
            ),
            legendgroup="route",
            customdata=[
                [i + 1, w.position.alt_m, w.heading, w.elapsed_min]
                for i, w in enumerate(plan.waypoints)
            ],
            hovertemplate=(
                "<b>WP %{customdata[0]}</b><br>"
                "%{lat:.4f}°, %{lon:.4f}°<br>"
                "%{customdata[1]:,.0f} m · heading %{customdata[2]:03.0f}°<br>"
                "T+%{customdata[3]:,.0f} min"
                "<extra></extra>"
            ),
        ),
    ]


def build_globe(
    hostile: Position,
    friendly: Position,
    defenses: list[DefenseSite] | None = None,
    plan=None,
    hostile_name: str = "RED",
    friendly_name: str = "BLUE",
) -> go.Figure:
    """Build the globe figure with both positions, the arc, defences and route."""
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
    # Rings first, then site markers, then the two principals — so nothing
    # important ends up underneath a threat envelope.
    for site in defenses or []:
        fig.add_trace(_threat_ring(site))
    if defenses:
        fig.add_trace(_defense_marker_trace(defenses))
    if plan is not None:
        for trace in _route_traces(plan):
            fig.add_trace(trace)

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
                + (f"<br>{plan.summary()}" if plan is not None else "")
                + "</span>"
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


CONTROL_STYLE = f"""
<style>
  #mission-controls {{
    position: fixed;
    top: 18px;
    right: 20px;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 8px;
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #mission-controls button {{
    padding: 9px 16px;
    border-radius: 6px;
    border: 1px solid {ROUTE_COLOR};
    background: rgba(62, 207, 117, 0.12);
    color: {ROUTE_COLOR};
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }}
  #mission-controls button:hover:not(:disabled) {{
    background: rgba(62, 207, 117, 0.22);
  }}
  #mission-controls button:disabled {{
    opacity: 0.5;
    cursor: progress;
  }}
  #mission-controls #new-scenario {{
    border-color: {TEXT_SECONDARY};
    background: rgba(166, 173, 184, 0.10);
    color: {TEXT_SECONDARY};
    font-weight: 500;
  }}
  #mission-status {{
    max-width: 320px;
    text-align: right;
    color: {TEXT_SECONDARY};
  }}
</style>
"""

CONTROL_HTML = """
<div id="mission-controls">
  <button id="plan-mission">Plan mission</button>
  <button id="new-scenario">New scenario</button>
  <div id="mission-status"></div>
</div>
"""

# The button posts to the server, which runs the actual PPO policy and returns
# ready-made traces. Inference stays in one place — the Python that was
# trained and evaluated — rather than being reimplemented here.
CONTROL_SCRIPT = """
(function () {
    var gd = document.getElementById('{plot_id}');
    var planBtn = document.getElementById('plan-mission');
    var newBtn = document.getElementById('new-scenario');
    var status = document.getElementById('mission-status');
    if (!gd || !planBtn) { return; }

    var routeIndices = [];

    planBtn.addEventListener('click', function () {
        planBtn.disabled = true;
        status.textContent = 'running policy…';

        fetch('/plan', {method: 'POST'})
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { status.textContent = data.error; return; }

                // Drop any previous route so repeated clicks do not stack up.
                var drop = routeIndices.length
                    ? Plotly.deleteTraces(gd, routeIndices)
                    : Promise.resolve();
                return drop.then(function () {
                    var first = gd.data.length;
                    routeIndices = data.traces.map(function (_, i) { return first + i; });
                    return Plotly.addTraces(gd, data.traces);
                }).then(function () {
                    status.textContent = data.summary;
                });
            })
            .catch(function (err) {
                status.textContent = 'plan failed: ' + err.message;
            })
            .finally(function () { planBtn.disabled = false; });
    });

    newBtn.addEventListener('click', function () {
        newBtn.disabled = true;
        status.textContent = 'generating…';
        fetch('/scenario', {method: 'POST'})
            .then(function () { window.location.reload(); })
            .catch(function (err) {
                status.textContent = 'failed: ' + err.message;
                newBtn.disabled = false;
            });
    });
})();
"""


def render_html(fig: go.Figure, interactive: bool = False) -> str:
    """Full-window HTML for the figure.

    `interactive` adds the mission-control buttons, which only work when the
    page is served by serve.py — a static file has no endpoint to call.
    """
    scripts = [HOVER_SCRIPT, FIT_SCRIPT]
    head = PAGE_STYLE
    if interactive:
        scripts.append(CONTROL_SCRIPT)
        head += CONTROL_STYLE

    html = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        post_script=scripts,
        config=GLOBE_CONFIG,
    )
    html = html.replace("</head>", f"{head}</head>", 1)
    if interactive:
        html = html.replace("<body>", f"<body>{CONTROL_HTML}", 1)
    return html


def write_globe(fig: go.Figure, path: Path) -> None:
    """Write the figure to a standalone, full-window HTML page."""
    Path(path).write_text(render_html(fig), encoding="utf-8")


def route_traces_json(plan) -> list[dict]:
    """Route traces as plain dicts, for the server to hand to Plotly.addTraces."""
    return [t.to_plotly_json() for t in _route_traces(plan)]
