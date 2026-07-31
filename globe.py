"""Renders tracked positions onto an interactive orthographic globe."""

from __future__ import annotations

from html import escape as html_escape
from pathlib import Path

import plotly.graph_objects as go

from defences import DefenceSite
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


def _marker_trace(pos: Position, name: str, colour: str, symbol: str) -> go.Scattergeo:
    return go.Scattergeo(
        lat=[pos.lat],
        lon=[pos.lon],
        name=name,
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
            line=dict(width=2.5, color=ROUTE_COLOUR if ok else ROUTE_FAILED_COLOUR),
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
                color=ROUTE_COLOUR if ok else ROUTE_FAILED_COLOUR,
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


def _ground_track_name(hostile: Position, friendly: Position) -> str:
    return f"Ground track · {great_circle_km(hostile, friendly):,.0f} km"


def _ground_track_trace(hostile: Position, friendly: Position) -> go.Scattergeo:
    """The dotted great-circle arc between the two principals.

    Hidden until a marker is hovered (see HOVER_SCRIPT). "legendonly" rather
    than False so the legend still names it and carries the range.
    """
    arc = great_circle_path(friendly, hostile)
    return go.Scattergeo(
        lat=[p.lat for p in arc],
        lon=[p.lon for p in arc],
        name=_ground_track_name(hostile, friendly),
        mode="lines",
        line=dict(width=2, color=ARC_COLOUR, dash="dot"),
        hoverinfo="skip",
        visible="legendonly",
    )


def ground_track_update(hostile: Position, friendly: Position) -> dict:
    """The arc as plain data, for redrawing it after BLUE has been dragged.

    Recomputed here rather than in the browser so there is one implementation
    of the great-circle interpolation.
    """
    arc = great_circle_path(friendly, hostile)
    return {
        "lat": [p.lat for p in arc],
        "lon": [p.lon for p in arc],
        "name": _ground_track_name(hostile, friendly),
    }


def title_text(
    hostile: Position,
    friendly: Position,
    plan=None,
    hostile_name: str = "RED",
    friendly_name: str = "BLUE",
) -> str:
    """The title block. Split out so the server can refresh it after a drag.

    Plotly does not wrap title text — a single long line runs past the paper's
    right edge and is clipped — so the readout is broken across explicit lines
    rather than left to fit.
    """
    return (
        "Tactical picture<br>"
        f"<span style='font-size:13px;color:{TEXT_SECONDARY}'>"
        f"{friendly_name} → {hostile_name}<br>"
        f"{slant_range_km(hostile, friendly):,.0f} km slant · "
        f"{great_circle_km(hostile, friendly):,.0f} km ground<br>"
        f"bearing {initial_bearing_deg(friendly, hostile):03.0f}° true · "
        f"elevation {elevation_angle_deg(friendly, hostile):+.2f}°"
        + (f"<br>{plan.summary()}" if plan is not None else "")
        + "</span>"
    )


def build_globe(
    hostile: Position,
    friendly: Position,
    defences: list[DefenceSite] | None = None,
    plan=None,
    hostile_name: str = "RED",
    friendly_name: str = "BLUE",
) -> go.Figure:
    """Build the globe figure with both positions, the arc, defences and route."""
    centre = midpoint(hostile, friendly)

    fig = go.Figure()

    # Trace 0, so the markers sit on top of it — and so ARC_TRACE_INDEX and the
    # drag handler's lookup both stay valid.
    fig.add_trace(_ground_track_trace(hostile, friendly))
    # Rings first, then site markers, then the two principals — so nothing
    # important ends up underneath a threat envelope.
    for site in defences or []:
        fig.add_trace(_threat_ring(site))
    if defences:
        fig.add_trace(_defence_marker_trace(defences))
    if plan is not None:
        for trace in _route_traces(plan):
            fig.add_trace(trace)

    fig.add_trace(_marker_trace(hostile, hostile_name, HOSTILE_COLOUR, "diamond"))
    fig.add_trace(_marker_trace(friendly, friendly_name, FRIENDLY_COLOUR, "circle"))

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
            text=title_text(hostile, friendly, plan, hostile_name, friendly_name),
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
  #callsign-row {{
    display: flex;
    align-items: center;
    gap: 6px;
    color: {TEXT_SECONDARY};
  }}
  #callsign-row input {{
    padding: 6px 8px;
    border-radius: 5px;
    border: 1px solid #39414d;
    background: rgba(255, 255, 255, 0.04);
    color: {TEXT_PRIMARY};
    font: inherit;
  }}
  #blue-number {{ width: 56px; }}
  #blue-callsign {{ width: 130px; }}
  #callsign-row button {{
    padding: 6px 12px;
    border-color: {FRIENDLY_COLOUR};
    background: rgba(57, 135, 229, 0.12);
    color: {FRIENDLY_COLOUR};
    font-weight: 500;
  }}
  #callsign-row button:hover:not(:disabled) {{
    background: rgba(57, 135, 229, 0.22);
  }}
  #mission-status {{
    max-width: 320px;
    text-align: right;
    color: {TEXT_SECONDARY};
  }}
</style>
"""

def control_html(blue_number: int = 1, blue_callsign: str = "") -> str:
    """Mission controls, pre-filled with the asset's current callsign."""
    callsign = html_escape(blue_callsign or "", quote=True)
    return f"""
<div id="mission-controls">
  <button id="plan-mission">Plan mission</button>
  <button id="new-scenario">New scenario</button>
  <div id="callsign-row">
    <label for="blue-number">BLUE</label>
    <input id="blue-number" type="number" min="1" step="1" value="{int(blue_number)}">
    <input id="blue-callsign" type="text" placeholder="callsign" value="{callsign}"
           maxlength="24">
    <button id="apply-callsign">Rename</button>
  </div>
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

    // ---- rename the friendly asset ---------------------------------------
    var applyBtn = document.getElementById('apply-callsign');
    var numberInput = document.getElementById('blue-number');
    var callsignInput = document.getElementById('blue-callsign');

    function applyCallsign() {
        applyBtn.disabled = true;
        fetch('/callsign', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                number: parseInt(numberInput.value, 10),
                callsign: callsignInput.value,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) { status.textContent = data.error; return; }
            var idx = blueIndex();
            if (idx >= 0) {
                // name drives the legend, text the on-map label, and the
                // hovertemplate has the name baked in — all three move together.
                Plotly.restyle(gd, {
                    name: [data.label],
                    text: [[data.label]],
                    hovertemplate: [data.hovertemplate],
                }, [idx]);
            }
            if (data.title) { Plotly.relayout(gd, {'title.text': data.title}); }
            numberInput.value = data.number;
            callsignInput.value = data.callsign;
            status.textContent = data.summary;
        })
        .catch(function (err) { status.textContent = 'rename failed: ' + err.message; })
        .finally(function () { applyBtn.disabled = false; });
    }

    if (applyBtn) {
        applyBtn.addEventListener('click', applyCallsign);
        [numberInput, callsignInput].forEach(function (el) {
            el.addEventListener('keydown', function (ev) {
                if (ev.key === 'Enter') { applyCallsign(); }
            });
        });
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
    var calibration = [0, 0];

    function subplot() {
        var fl = gd._fullLayout;
        return fl && fl.geo && fl.geo._subplot;
    }

    // Prefix match, not equality: the trace is named "BLUE 2 (Viper)" once a
    // callsign is set, and an exact test would silently stop finding it.
    function blueIndex() {
        for (var i = 0; i < gd.data.length; i++) {
            if ((gd.data[i].name || '').indexOf('BLUE') === 0) { return i; }
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
        var idx = blueIndex();
        if (idx < 0) { return; }

        var hit = onMarker(ev, idx);
        if (!hit) { return; }

        calibration = [0, 0];
        var screen = toScreen(gd.data[idx].lon[0], gd.data[idx].lat[0]);
        if (hit === 'dom' && screen) {
            calibration = [ev.clientX - screen[0], ev.clientY - screen[1]];
        }

        dragging = true;
        gd.style.cursor = 'grabbing';
        status.textContent = 'moving BLUE…';
        ev.preventDefault();
        ev.stopPropagation();
    }, true);

    // Each restyle redraws the geo layer, so coalesce to one per frame instead
    // of one per mousemove event.
    var queued = null;

    function flush() {
        queued = null;
        if (!dragging || !pending) { return; }
        var idx = blueIndex();
        if (idx >= 0) {
            Plotly.restyle(gd, {lon: [[pending[0]]], lat: [[pending[1]]]}, [idx]);
        }
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

        var idx = blueIndex();
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
                body: JSON.stringify({lat: lat, lon: lon}),
            });
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.title) { Plotly.relayout(gd, {'title.text': data.title}); }

            // The arc ran from where BLUE used to be; the server sends back a
            // recomputed one, along with a legend label carrying the new range.
            if (data.groundTrack) {
                var gi = -1;
                for (var i = 0; i < gd.data.length; i++) {
                    if ((gd.data[i].name || '').indexOf('Ground track') === 0) {
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
    blue_number: int = 1,
    blue_callsign: str = "",
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
        html = html.replace(
            "<body>", f"<body>{control_html(blue_number, blue_callsign)}", 1
        )
    return html


def write_globe(fig: go.Figure, path: Path) -> None:
    """Write the figure to a standalone, full-window HTML page."""
    Path(path).write_text(render_html(fig), encoding="utf-8")


def route_traces_json(plan) -> list[dict]:
    """Route traces as plain dicts, for the server to hand to Plotly.addTraces."""
    return [t.to_plotly_json() for t in _route_traces(plan)]
