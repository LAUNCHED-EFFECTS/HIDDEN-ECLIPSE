"""Renders tracked positions onto an interactive orthographic globe."""

from __future__ import annotations

import math
from html import escape as html_escape
from pathlib import Path

import plotly.graph_objects as go

from defences import DefenceSite
from geo import (
    EARTH_RADIUS_KM,
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
# reads without colour: hostile is a red diamond, friendly a blue circle.
HOSTILE_COLOUR = "#e66767"
FRIENDLY_COLOUR = "#3987e5"
ARC_COLOUR = "#8a8a82"

# Defence rings share the hostile hue but sit well back from it in weight, so
# the target marker still reads as the brightest thing on the map.
THREAT_LINE = "#b8524f"
THREAT_FILL = "rgba(230, 103, 103, 0.10)"

# Green on success, amber on failure — so the outcome is legible from the map
# alone, not only from the title. Green also keeps the route distinct from
# BLUE's own marker, which the old blue route sat right on top of.
ROUTE_COLOUR = "#3ecf75"
ROUTE_FAILED_COLOUR = "#eda100"

OCEAN = "#12161c"
LAND = "#242a33"
COASTLINE = "#4a525e"
SURFACE = "#0d1117"
TEXT_PRIMARY = "#f0f2f5"
TEXT_SECONDARY = "#a6adb8"


def marker_hovertemplate(name: str) -> str:
    """Hover text for a principal's marker.

    The name is baked into the template, so renaming an asset has to restyle
    this too — hence it being shared rather than inlined.
    """
    return (
        f"<b>{name}</b><br>"
        "%{lat:.4f}°, %{lon:.4f}°<br>"
        "altitude %{customdata[0]:,.0f} m (%{customdata[1]:,.0f} ft)"
        "<extra></extra>"
    )


def _marker_trace(
    pos: Position, name: str, colour: str, symbol: str, meta: dict | None = None
) -> go.Scattergeo:
    """A principal's marker.

    `meta` tags the trace so the browser can find it by role rather than by
    name. Name matching is not safe here: once a mission is planned the route
    traces are called "BLUE 2 · 1,245 km" and a prefix search finds those first.
    """
    return go.Scattergeo(
        lat=[pos.lat],
        lon=[pos.lon],
        name=name,
        meta=meta,
        mode="markers+text",
        marker=dict(
            size=14,
            # `color` is plotly's own keyword, so it keeps the US spelling here.
            color=colour,
            symbol=symbol,
            # 2px surface ring keeps the mark legible against land or ocean.
            line=dict(width=2, color=SURFACE),
        ),
        text=[name],
        textposition="top center",
        textfont=dict(size=12, color=TEXT_PRIMARY),
        customdata=[[pos.alt_m, pos.alt_ft]],
        hovertemplate=marker_hovertemplate(name),
    )


def _threat_ring(site: DefenceSite) -> go.Scattergeo:
    """A filled circle at the site's engagement radius."""
    ring = circle_path(site.position, site.kind.engagement_km)
    return go.Scattergeo(
        lat=[p.lat for p in ring],
        lon=[p.lon for p in ring],
        mode="lines",
        line=dict(width=1, color=THREAT_LINE),
        fill="toself",
        fillcolor=THREAT_FILL,
        legendgroup="defences",
        showlegend=False,
        hoverinfo="skip",  # the site marker carries the detail
    )


def _defence_marker_trace(sites: list[DefenceSite]) -> go.Scattergeo:
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
        legendgroup="defences",
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


def _route_traces(package) -> list[go.Scattergeo]:
    """Every asset's route and waypoints — two traces per aircraft.

    Each leg is coloured by that aircraft's own fate, so a package where one
    got through and another was lost reads correctly on the map instead of
    collapsing into a single verdict.
    """
    traces = []
    for asset in package.assets:
        if len(asset.track) < 2:
            continue
        colour = ROUTE_COLOUR if asset.succeeded else ROUTE_FAILED_COLOUR
        group = f"route-{asset.label}"

        traces.append(
            go.Scattergeo(
                lat=[p.lat for p in asset.track],
                lon=[p.lon for p in asset.track],
                name=f"{asset.label} · {asset.route_km:,.0f} km",
                mode="lines",
                line=dict(width=2.5, color=colour),
                legendgroup=group,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scattergeo(
                lat=[w.position.lat for w in asset.waypoints],
                lon=[w.position.lon for w in asset.waypoints],
                name=f"{asset.label} waypoints",
                mode="markers",
                marker=dict(
                    size=6,
                    color=colour,
                    symbol="circle",
                    line=dict(width=1, color=SURFACE),
                ),
                legendgroup=group,
                showlegend=False,
                customdata=[
                    [i + 1, w.position.alt_m, w.heading, w.elapsed_min]
                    for i, w in enumerate(asset.waypoints)
                ],
                hovertemplate=(
                    f"<b>{asset.label} WP %{{customdata[0]}}</b><br>"
                    "%{lat:.4f}°, %{lon:.4f}°<br>"
                    "%{customdata[1]:,.0f} m · heading %{customdata[2]:03.0f}°<br>"
                    "T+%{customdata[3]:,.0f} min"
                    "<extra></extra>"
                ),
            )
        )
    return traces


# How far out toward the viewport edge the further principal is allowed to sit.
# 1.0 would put it exactly on the edge; this leaves a margin for its label.
FIT_MARGIN = 0.78


def fit_scale_limit(hostile: Position, friendly: Position) -> float:
    """Largest projection.scale that still shows both principals.

    An orthographic projection puts a point at angular distance α from the
    camera centre at a screen radius of R·sin(α), where R is half the globe's
    rendered size. The camera sits on the midpoint, so each principal is α = θ/2
    away for a central angle θ — and keeping both inside the viewport means

        scale · (min(w, h) / 2) · sin(θ/2)  ≤  margin · min(w, h) / 2

    which reduces to scale ≤ margin / sin(θ/2), independent of window size.
    Using min(w, h) is the conservative choice: the on-screen bearing between
    the two is unknown, so the shorter axis has to be assumed.
    """
    theta = great_circle_km(hostile, friendly) / EARTH_RADIUS_KM  # radians
    return FIT_MARGIN / max(math.sin(theta / 2), 1e-9)


def _ground_track_name(hostile: Position, friendly: Position, label: str) -> str:
    return f"{label} ground track · {great_circle_km(hostile, friendly):,.0f} km"


def _ground_track_trace(
    hostile: Position, friendly: Position, label: str, index: int
) -> go.Scattergeo:
    """One aircraft's dotted great-circle arc to the target.

    Hidden until its marker is hovered — or until RED is, which reveals the
    whole set (see HOVER_SCRIPT). "legendonly" rather than False so the legend
    still names it and carries the range.
    """
    arc = great_circle_path(friendly, hostile)
    return go.Scattergeo(
        lat=[p.lat for p in arc],
        lon=[p.lon for p in arc],
        name=_ground_track_name(hostile, friendly, label),
        meta={"role": "track", "asset": index},
        mode="lines",
        line=dict(width=2, color=ARC_COLOUR, dash="dot"),
        hoverinfo="skip",
        visible="legendonly",
    )


def ground_track_update(
    hostile: Position, friendly: Position, label: str, index: int
) -> dict:
    """One arc as plain data, for redrawing it after that aircraft was dragged.

    Recomputed here rather than in the browser so there is one implementation
    of the great-circle interpolation.
    """
    arc = great_circle_path(friendly, hostile)
    return {
        "asset": index,
        "lat": [p.lat for p in arc],
        "lon": [p.lon for p in arc],
        "name": _ground_track_name(hostile, friendly, label),
    }


def title_text(
    hostile: Position,
    friendly: Position,
    plan=None,
    hostile_name: str = "RED",
    friendly_name: str = "BLUE",
    package_size: int = 1,
) -> str:
    """The title block. Split out so the server can refresh it after a drag.

    Plotly does not wrap title text — a single long line runs past the paper's
    right edge and is clipped — so the readout is broken across explicit lines
    rather than left to fit.
    """
    return (
        "Tactical picture<br>"
        f"<span style='font-size:13px;color:{TEXT_SECONDARY}'>"
        f"{friendly_name}{f' +{package_size - 1}' if package_size > 1 else ''}"
        f" → {hostile_name}<br>"
        f"{slant_range_km(hostile, friendly):,.0f} km slant · "
        f"{great_circle_km(hostile, friendly):,.0f} km ground<br>"
        f"bearing {initial_bearing_deg(friendly, hostile):03.0f}° true · "
        f"elevation {elevation_angle_deg(friendly, hostile):+.2f}°"
        + (f"<br>{plan.summary()}" if plan is not None else "")
        + "</span>"
    )


def build_globe(
    hostile: Position,
    friendlies: list[Position] | Position,
    defences: list[DefenceSite] | None = None,
    plan=None,
    hostile_name: str = "RED",
    friendly_names: list[str] | str | None = None,
) -> go.Figure:
    """Build the globe: the package, RED, the defences, and any planned routes.

    `friendlies` accepts a bare Position for the single-asset case, so callers
    that predate the package do not have to wrap it.
    """
    if isinstance(friendlies, Position):
        friendlies = [friendlies]
    if friendly_names is None:
        friendly_names = [f"BLUE {i + 1}" for i in range(len(friendlies))]
    elif isinstance(friendly_names, str):
        friendly_names = [friendly_names]

    lead = friendlies[0]
    # The ground track and the camera framing both reference the lead asset.
    centre = midpoint(hostile, lead)

    fig = go.Figure()

    # Tracks first so every marker sits on top of them. One per aircraft, each
    # tagged with its index so the hover logic can reveal exactly one — or all
    # of them when RED is hovered.
    for i, (position, name) in enumerate(zip(friendlies, friendly_names)):
        fig.add_trace(_ground_track_trace(hostile, position, name, i))
    # Rings first, then site markers, then the two principals — so nothing
    # important ends up underneath a threat envelope.
    for site in defences or []:
        fig.add_trace(_threat_ring(site))
    if defences:
        fig.add_trace(_defence_marker_trace(defences))
    if plan is not None:
        for trace in _route_traces(plan):
            fig.add_trace(trace)

    fig.add_trace(
        _marker_trace(
            hostile, hostile_name, HOSTILE_COLOUR, "diamond", meta={"role": "hostile"}
        )
    )
    for i, (position, name) in enumerate(zip(friendlies, friendly_names)):
        fig.add_trace(
            _marker_trace(
                position,
                name,
                FRIENDLY_COLOUR,
                "circle",
                meta={"role": "asset", "asset": i},
            )
        )

    fig.update_geos(
        projection=dict(
            type="orthographic",
            # Aim the camera at the midpoint so both marks face the viewer
            # whenever they are on the same hemisphere.
            rotation=dict(lat=centre.lat, lon=centre.lon, roll=0),
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
            text=title_text(
                hostile,
                lead,
                plan,
                hostile_name,
                friendly_names[0],
                package_size=len(friendlies),
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
        # Read by FIT_SCRIPT in the browser, which is where the window's aspect
        # ratio — the other half of the zoom calculation — is known.
        meta=dict(fitMaxScale=min(
            fit_scale_limit(hostile, f) for f in friendlies
        )),
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


# Hover-driven visibility is not expressible in plotly's declarative API, so it
# rides along in the exported HTML as a post-plot hook.
#
# Hovering an aircraft reveals that aircraft's track; hovering RED reveals every
# track at once. Traces are found by meta.role rather than by index, because the
# trace list changes with the package size and with whether a route is drawn.
HOVER_SCRIPT = """
(function () {
    var gd = document.getElementById('{plot_id}');
    var shown = [];
    var pending = null;

    function trackTraces() {
        var byAsset = {}, all = [];
        for (var i = 0; i < gd.data.length; i++) {
            var m = gd.data[i].meta;
            if (m && m.role === 'track') { byAsset[m.asset] = i; all.push(i); }
        }
        return {byAsset: byAsset, all: all};
    }

    function sameSet(a, b) {
        if (a.length !== b.length) { return false; }
        for (var i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) { return false; }
        }
        return true;
    }

    function set(wanted) {
        if (pending) { clearTimeout(pending); pending = null; }
        wanted = wanted.slice().sort(function (x, y) { return x - y; });
        if (sameSet(wanted, shown)) { return; }   // never restyle a no-op: each
                                                  // one redraws the globe

        var hide = shown.filter(function (i) { return wanted.indexOf(i) < 0; });
        if (hide.length) { Plotly.restyle(gd, {visible: 'legendonly'}, hide); }
        if (wanted.length) { Plotly.restyle(gd, {visible: true}, wanted); }
        shown = wanted;
    }

    gd.on('plotly_hover', function (ev) {
        var point = ev && ev.points && ev.points[0];
        if (!point) { return; }
        var meta = gd.data[point.curveNumber] && gd.data[point.curveNumber].meta;
        var tracks = trackTraces();

        if (meta && meta.role === 'hostile') {
            set(tracks.all);                       // everything converging on RED
        } else if (meta && meta.role === 'asset') {
            var one = tracks.byAsset[meta.asset];
            set(one === undefined ? [] : [one]);
        } else {
            set([]);                               // defence sites, waypoints
        }
    });

    // Deferred, so sliding the cursor straight from one marker to the next does
    // not blink the track off between them. The redraw from set() can itself
    // emit an unhover, which this delay also absorbs.
    gd.on('plotly_unhover', function () {
        pending = setTimeout(function () { pending = null; set([]); }, 120);
    });
})();
"""


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

        // Filling the width zooms in, which can push one of the principals off
        // screen when they are far apart. layout.meta carries the largest scale
        // that still shows both; it wins, and may pull below 1 (i.e. zoom out
        // past the default fit) when they are nearly antipodal.
        var meta = gd._fullLayout.meta || {};
        var limit = meta.fitMaxScale || Infinity;
        var scale = Math.min(Math.max(1, FILL * size.w / size.h), limit);

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
    /* Clear of plotly's modebar, which is right-aligned at the top of the
       plot area — at 18px the buttons sat underneath it. */
    top: 64px;
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
    border: 1px solid {ROUTE_COLOUR};
    background: rgba(62, 207, 117, 0.12);
    color: {ROUTE_COLOUR};
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
  #asset-list {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .asset-row {{
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 6px;
    color: {TEXT_SECONDARY};
  }}
  .asset-label {{
    min-width: 128px;
    text-align: right;
    color: {TEXT_PRIMARY};
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: pointer;
  }}
  .asset-label:hover {{
    color: {FRIENDLY_COLOUR};
    text-decoration: underline;
  }}
  .asset-row input {{
    padding: 6px 8px;
    border-radius: 5px;
    border: 1px solid #39414d;
    background: rgba(255, 255, 255, 0.04);
    color: {TEXT_PRIMARY};
    font: inherit;
  }}
  .asset-number {{ width: 56px; }}
  .asset-callsign {{ width: 120px; }}
  .asset-row button {{
    padding: 6px 10px;
    border-color: {FRIENDLY_COLOUR};
    background: rgba(57, 135, 229, 0.12);
    color: {FRIENDLY_COLOUR};
    font-weight: 500;
  }}
  .asset-row button:hover:not(:disabled) {{
    background: rgba(57, 135, 229, 0.22);
  }}
  .asset-remove {{
    min-width: 32px;
    border-color: {ROUTE_FAILED_COLOUR} !important;
    background: rgba(237, 161, 0, 0.12) !important;
    color: {ROUTE_FAILED_COLOUR} !important;
  }}
  .asset-remove:hover:not(:disabled) {{
    background: rgba(237, 161, 0, 0.22) !important;
  }}
  #add-asset {{
    border-color: {TEXT_SECONDARY};
    background: rgba(166, 173, 184, 0.10);
    color: {TEXT_SECONDARY};
    font-weight: 500;
  }}
  #add-asset:hover:not(:disabled) {{
    background: rgba(166, 173, 184, 0.20);
  }}
  #mission-status {{
    max-width: 320px;
    text-align: right;
    color: {TEXT_SECONDARY};
  }}
</style>
"""

def control_html(assets: list[dict] | None = None) -> str:
    """Mission controls, with every aircraft in the package listed at once.

    Each row owns its aircraft: the fields are pre-filled from it, and the
    buttons act on the index in its `data-asset` attribute — so nothing has to
    track a current selection.
    """
    assets = assets or [{"number": 1, "callsign": "", "label": "BLUE 1"}]
    rows = "\n".join(
        f"""    <div class="asset-row" data-asset="{i}">
      <span class="asset-label" title="centre the globe on this aircraft"
            >{html_escape(a["label"])}</span>
      <input class="asset-number" type="number" min="1" step="1"
             value="{int(a["number"])}" title="asset number">
      <input class="asset-callsign" type="text" placeholder="callsign"
             value="{html_escape(a["callsign"] or "", quote=True)}" maxlength="24">
      <button class="asset-rename" title="apply this number and callsign">Rename</button>
      <button class="asset-remove" title="remove this aircraft">&minus;</button>
    </div>"""
        for i, a in enumerate(assets)
    )
    return f"""
<div id="mission-controls">
  <button id="plan-mission">Plan mission</button>
  <button id="new-scenario">New scenario</button>
  <div id="asset-list">
{rows}
  </div>
  <button id="add-asset">+ Add aircraft</button>
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

    // ---- rename / add / remove assets ------------------------------------
    //
    // Every aircraft has its own row, so the handlers are delegated off the
    // list and read the index from the row's data-asset attribute. Nothing
    // tracks a "current" aircraft.
    var assetList = document.getElementById('asset-list');
    var addBtn = document.getElementById('add-asset');

    function rowIndex(row) {
        return parseInt(row.getAttribute('data-asset'), 10);
    }

    // Clicking an aircraft's name swings the camera onto it. Rotating the
    // projection is what "centre" means on an orthographic globe — panning
    // would slide the map instead of turning the sphere.
    function centreOn(index) {
        var trace = traceForAsset(index);
        if (trace < 0) { return; }
        var lat = gd.data[trace].lat[0];
        var lon = gd.data[trace].lon[0];
        Plotly.relayout(gd, {
            'geo.projection.rotation.lat': lat,
            'geo.projection.rotation.lon': lon,
        });
        status.textContent = 'centred on ' + (gd.data[trace].name || 'aircraft');
    }

    function renameFrom(row) {
        var index = rowIndex(row);
        var numberInput = row.querySelector('.asset-number');
        var callsignInput = row.querySelector('.asset-callsign');
        var button = row.querySelector('.asset-rename');

        button.disabled = true;
        fetch('/callsign', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                index: index,
                number: parseInt(numberInput.value, 10),
                callsign: callsignInput.value,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) { status.textContent = data.error; return; }

            var trace = traceForAsset(index);
            if (trace >= 0) {
                // name drives the legend, text the on-map label, and the
                // hovertemplate has the name baked in — all three move together.
                Plotly.restyle(gd, {
                    name: [data.label],
                    text: [[data.label]],
                    hovertemplate: [data.hovertemplate],
                }, [trace]);
            }
            if (data.title) { Plotly.relayout(gd, {'title.text': data.title}); }

            row.querySelector('.asset-label').textContent = data.label;
            numberInput.value = data.number;
            callsignInput.value = data.callsign;
            status.textContent = data.summary;
        })
        .catch(function (err) { status.textContent = 'rename failed: ' + err.message; })
        .finally(function () { button.disabled = false; });
    }

    // Adding or removing changes the trace list wholesale, so the page is
    // rebuilt rather than patched — far less to get wrong than splicing traces
    // and re-indexing everything that refers to them.
    function editPackage(action, index) {
        status.textContent = action === 'add' ? 'adding…' : 'removing…';
        fetch('/assets', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: action, index: index}),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) { status.textContent = data.error; return; }
            window.location.reload();
        })
        .catch(function (err) { status.textContent = 'failed: ' + err.message; });
    }

    if (assetList) {
        assetList.addEventListener('click', function (ev) {
            var row = ev.target.closest ? ev.target.closest('.asset-row') : null;
            if (!row) { return; }
            if (ev.target.classList.contains('asset-rename')) { renameFrom(row); }
            else if (ev.target.classList.contains('asset-remove')) {
                editPackage('remove', rowIndex(row));
            }
            else if (ev.target.classList.contains('asset-label')) {
                centreOn(rowIndex(row));
            }
        });
        assetList.addEventListener('keydown', function (ev) {
            if (ev.key !== 'Enter') { return; }
            var row = ev.target.closest ? ev.target.closest('.asset-row') : null;
            if (row) { renameFrom(row); }
        });
    }

    if (addBtn) {
        addBtn.addEventListener('click', function () { editPackage('add', 0); });
    }

    // ---- drag BLUE to reposition it -------------------------------------
    //
    // The projection maps lon/lat to *paper* coordinates — plotly's own geo
    // code inverts with `[x + xaxis._offset, y + yaxis._offset]`, i.e. relative
    // to the plot origin, so client coords convert by subtracting the div's
    // top-left. A calibration offset is measured on mousedown anyway, since
    // that press is known to be on the marker: if the assumption above is ever
    // off by a constant, this cancels it out for the rest of the drag.
    var HIT_RADIUS = 18;
    var dragging = false;
    var dragTrace = -1;      // plotly trace being dragged
    var dragAsset = -1;      // that trace's index within the package
    var calibration = [0, 0];

    function subplot() {
        var fl = gd._fullLayout;
        return fl && fl.geo && fl.geo._subplot;
    }

    // Marker traces are tagged with meta.role, because names are ambiguous:
    // after planning, "BLUE 2 · 1,245 km" is a route and "BLUE 2 waypoints" is
    // its turn points, and a name search would find those instead.
    function assetTraces() {
        var found = [];
        for (var i = 0; i < gd.data.length; i++) {
            var m = gd.data[i].meta;
            if (m && m.role === 'asset') { found.push({trace: i, asset: m.asset}); }
        }
        return found;
    }

    function traceForAsset(assetIndex) {
        var all = assetTraces();
        for (var i = 0; i < all.length; i++) {
            if (all[i].asset === assetIndex) { return all[i].trace; }
        }
        return -1;
    }

    function toScreen(lon, lat) {
        var sp = subplot();
        if (!sp || !sp.projection) { return null; }
        var p = sp.projection([lon, lat]);
        if (!p) { return null; }                  // on the far side of the globe
        var r = gd.getBoundingClientRect();
        return [p[0] + r.left, p[1] + r.top];
    }

    function toLonLat(clientX, clientY) {
        var sp = subplot();
        if (!sp || !sp.projection) { return null; }
        var r = gd.getBoundingClientRect();
        return sp.projection.invert([
            clientX - r.left - calibration[0],
            clientY - r.top - calibration[1],
        ]);
    }

    function onMarker(ev, idx) {
        // DOM test first — exact, and independent of any coordinate maths.
        var groups = gd.querySelectorAll('.trace.scattergeo');
        if (groups.length === gd.data.length && groups[idx] &&
            groups[idx].contains(ev.target)) {
            return 'dom';
        }
        var screen = toScreen(gd.data[idx].lon[0], gd.data[idx].lat[0]);
        if (!screen) { return null; }
        var dx = ev.clientX - screen[0], dy = ev.clientY - screen[1];
        return Math.sqrt(dx * dx + dy * dy) <= HIT_RADIUS ? 'geometric' : null;
    }

    // Capture phase: this has to beat the globe's own rotate-drag handler,
    // which is bound to a descendant of gd.
    gd.addEventListener('mousedown', function (ev) {
        // Any asset is draggable, so test them all and take the nearest hit
        // rather than assuming the lead.
        var best = null, bestDist = Infinity;
        var candidates = assetTraces();

        for (var i = 0; i < candidates.length; i++) {
            var idx = candidates[i].trace;
            var hit = onMarker(ev, idx);
            if (!hit) { continue; }
            var screen = toScreen(gd.data[idx].lon[0], gd.data[idx].lat[0]);
            var dist = 0;
            if (screen) {
                var dx = ev.clientX - screen[0], dy = ev.clientY - screen[1];
                dist = Math.sqrt(dx * dx + dy * dy);
            }
            if (dist < bestDist) {
                bestDist = dist;
                best = {trace: idx, asset: candidates[i].asset, hit: hit, screen: screen};
            }
        }
        if (!best) { return; }

        calibration = [0, 0];
        if (best.hit === 'dom' && best.screen) {
            calibration = [ev.clientX - best.screen[0], ev.clientY - best.screen[1]];
        }

        dragging = true;
        dragTrace = best.trace;
        dragAsset = best.asset;
        gd.style.cursor = 'grabbing';
        status.textContent = 'moving ' + (gd.data[best.trace].name || 'asset') + '…';
        ev.preventDefault();
        ev.stopPropagation();
    }, true);

    // Each restyle redraws the geo layer, so coalesce to one per frame instead
    // of one per mousemove event.
    var queued = null;

    function flush() {
        queued = null;
        if (!dragging || !pending || dragTrace < 0) { return; }
        Plotly.restyle(gd, {lon: [[pending[0]]], lat: [[pending[1]]]}, [dragTrace]);
    }

    var pending = null;

    window.addEventListener('mousemove', function (ev) {
        if (!dragging) { return; }
        var lonlat = toLonLat(ev.clientX, ev.clientY);
        // invert() returns null past the limb of the globe — hold the last
        // good position rather than snapping somewhere arbitrary.
        if (!lonlat) { return; }
        pending = lonlat;
        if (queued === null) { queued = window.requestAnimationFrame(flush); }
        ev.preventDefault();
    }, true);

    window.addEventListener('mouseup', function (ev) {
        if (!dragging) { return; }
        dragging = false;
        gd.style.cursor = '';

        var idx = dragTrace, asset = dragAsset;
        dragTrace = -1;
        if (idx < 0) { return; }

        // A frame may still be queued; apply it so the committed position is
        // where the cursor actually was, not one frame behind.
        if (queued !== null) { window.cancelAnimationFrame(queued); queued = null; }
        if (pending) {
            Plotly.restyle(gd, {lon: [[pending[0]]], lat: [[pending[1]]]}, [idx]);
        }
        var lat = pending ? pending[1] : gd.data[idx].lat[0];
        var lon = pending ? pending[0] : gd.data[idx].lon[0];
        pending = null;

        // Any existing route was flown from the old start point.
        var drop = routeIndices.length
            ? Plotly.deleteTraces(gd, routeIndices)
            : Promise.resolve();
        routeIndices = [];

        drop.then(function () {
            return fetch('/blue', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: asset, lat: lat, lon: lon}),
            });
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.title) { Plotly.relayout(gd, {'title.text': data.title}); }

            // That aircraft's arc ran from where it used to be; the server
            // sends back a recomputed one carrying the new range. Only its own
            // track moves — the rest of the package has not.
            if (data.groundTrack) {
                var gi = -1;
                for (var i = 0; i < gd.data.length; i++) {
                    var tm = gd.data[i].meta;
                    if (tm && tm.role === 'track' && tm.asset === data.groundTrack.asset) {
                        gi = i;
                        break;
                    }
                }
                if (gi >= 0) {
                    Plotly.restyle(gd, {
                        lat: [data.groundTrack.lat],
                        lon: [data.groundTrack.lon],
                        name: [data.groundTrack.name],
                    }, [gi]);
                }
            }

            status.textContent = data.summary || 'BLUE moved — plan again';
        })
        .catch(function (err) { status.textContent = 'move failed: ' + err.message; });
    }, true);
})();
"""


def render_html(
    fig: go.Figure,
    interactive: bool = False,
    assets: list[dict] | None = None,
) -> str:
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
        html = html.replace("<body>", f"<body>{control_html(assets)}", 1)
    return html


def write_globe(fig: go.Figure, path: Path) -> None:
    """Write the figure to a standalone, full-window HTML page."""
    Path(path).write_text(render_html(fig), encoding="utf-8")


def route_traces_json(plan) -> list[dict]:
    """Route traces as plain dicts, for the server to hand to Plotly.addTraces."""
    return [t.to_plotly_json() for t in _route_traces(plan)]
