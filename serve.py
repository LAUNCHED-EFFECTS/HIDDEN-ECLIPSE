"""Serve the globe with working mission-control buttons.

    python3 serve.py                       # http://127.0.0.1:8000
    python3 serve.py --port 9000 --seed 21

The page itself is the same figure the CLI writes; the difference is that the
buttons have somewhere to call. Planning runs the trained PyTorch policy in
this process, so there is exactly one implementation of the environment and
the network — the one that was trained and evaluated.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from geo import Position, great_circle_km
from globe import (
    build_globe,
    ground_track_update,
    marker_hovertemplate,
    render_html,
    route_traces_json,
    title_text,
)
from plan import plan_mission
from ppo import load_policy
from world import World, clean_callsign, generate_world


class MissionState:
    """The scenario currently on screen, plus the loaded policy.

    Guarded by a lock because ThreadingHTTPServer handles requests
    concurrently and a click can land while another is still planning.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.rng = random.Random(args.seed)
        self.model = None
        self.norm = None

        if args.policy.exists():
            self.model, self.norm = load_policy(args.policy)
        self.world = self._generate()

    def _generate(self) -> World:
        a = self.args
        return generate_world(
            self.rng,
            tuple(a.lat_range),
            tuple(a.lon_range),
            tuple(a.alt_range),
            a.defences,
            a.defence_spread,
            a.blue_number,
            a.blue_callsign,
            a.blues,
        )

    def new_scenario(self) -> None:
        with self.lock:
            # Call signs belong to the aircraft, not the scenario — a fresh
            # laydown should not silently rename what the user just named. The
            # package size carries over too.
            identities = [(a.number, a.callsign) for a in self.world.assets]
            self.world = self._generate()
            while len(self.world.assets) < len(identities):
                self.world.add_asset(self.rng)
            self.world.assets = self.world.assets[: len(identities)]
            for asset, (number, callsign) in zip(self.world.assets, identities):
                asset.number, asset.callsign = number, callsign

    def page(self) -> str:
        with self.lock:
            world = self.world
        fig = build_globe(
            world.hostile,
            world.friendlies,
            world.defences,
            friendly_names=world.friendly_labels,
        )
        return render_html(
            fig,
            interactive=True,
            assets=[
                {"number": a.number, "callsign": a.callsign, "label": a.label}
                for a in world.assets
            ],
        )

    def set_callsign(self, index, number, callsign) -> dict:
        """Renumber or rename one aircraft in the package."""
        try:
            number = max(1, int(number))
            index = int(index)
        except (TypeError, ValueError):
            return {"error": "asset number must be a whole number"}

        with self.lock:
            if not 0 <= index < len(self.world.assets):
                return {"error": "no such asset"}
            asset = self.world.assets[index]
            asset.number = number
            asset.callsign = clean_callsign(callsign)
            world, label = self.world, asset.label

        return {
            "index": index,
            "label": label,
            "number": asset.number,
            "callsign": asset.callsign,
            "hovertemplate": marker_hovertemplate(label),
            "title": title_text(
                world.hostile,
                world.lead,
                friendly_name=world.assets[0].label,
                package_size=len(world.assets),
            ),
            "summary": f"renamed to {label}",
        }

    def edit_package(self, action: str, index: int) -> dict:
        """Add an aircraft to the package, or remove the selected one."""
        with self.lock:
            if action == "add":
                asset = self.world.add_asset(self.rng)
                return {
                    "ok": True,
                    "count": len(self.world.assets),
                    "summary": f"added {asset.label}",
                }
            if action == "remove":
                if len(self.world.assets) <= 1:
                    return {"error": "the package needs at least one aircraft"}
                if not self.world.remove_asset(int(index)):
                    return {"error": "no such asset"}
                return {"ok": True, "count": len(self.world.assets)}
        return {"error": f"unknown action {action!r}"}

    def move_blue(self, index: int, lat: float, lon: float) -> dict:
        """Reposition one aircraft after a drag, keeping its altitude."""
        with self.lock:
            if not 0 <= int(index) < len(self.world.assets):
                return {"error": "no such asset"}
            asset = self.world.assets[int(index)]
            asset.position = Position(float(lat), float(lon), asset.position.alt_m)
            world, label = self.world, asset.label

        return {
            "title": title_text(
                world.hostile,
                world.lead,
                friendly_name=world.assets[0].label,
                package_size=len(world.assets),
            ),
            "groundTrack": ground_track_update(
                world.hostile, asset.position, label, int(index)
            ),
            "summary": (
                f"{label} at {asset.position.coords} · "
                f"{great_circle_km(asset.position, world.hostile):,.0f} km to RED"
                " — plan again"
            ),
        }

    def plan(self) -> dict:
        if self.model is None:
            return {"error": f"no policy at {self.args.policy} — run train.py first"}

        with self.lock:
            scenario = self.world.to_scenario()
            labels = self.world.friendly_labels

        mission = plan_mission(scenario, self.model, self.norm, labels=labels)
        payload = {
            "summary": mission.summary(),
            "succeeded": mission.succeeded,
            "traces": route_traces_json(mission),
        }
        # Closest approach is per aircraft now; report the tightest one, since
        # that is the leg that came nearest to being engaged.
        margins = [
            a for a in mission.assets
            if a.threatening_site and a.closest_margin_km == a.closest_margin_km
        ]
        if margins:
            tightest = min(margins, key=lambda a: a.closest_margin_km)
            gap = tightest.closest_margin_km
            edge = "inside" if gap < 0 else "clear of"
            payload["summary"] += (
                f" · {tightest.label} came {abs(gap):,.1f} km {edge} "
                f"{tightest.threatening_site}"
            )
        return payload


def make_handler(state: MissionState):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page is regenerated per request; caching it would hide a
            # freshly generated scenario behind the previous one.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(state.page().encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:
            if self.path == "/plan":
                payload = state.plan()
            elif self.path == "/scenario":
                state.new_scenario()
                payload = {"ok": True}
            elif self.path in ("/blue", "/callsign", "/assets"):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if self.path == "/blue":
                        payload = state.move_blue(
                            body.get("index", 0), body["lat"], body["lon"]
                        )
                    elif self.path == "/assets":
                        payload = state.edit_package(
                            body.get("action", ""), body.get("index", 0)
                        )
                    else:
                        payload = state.set_callsign(
                            body.get("index", 0),
                            body.get("number", 1),
                            body.get("callsign", ""),
                        )
                except (ValueError, KeyError, TypeError) as exc:
                    self._send(
                        json.dumps({"error": f"bad request: {exc}"}).encode(),
                        "application/json",
                        400,
                    )
                    return
            else:
                self._send(b"not found", "text/plain", 404)
                return
            self._send(json.dumps(payload).encode("utf-8"), "application/json")

        def log_message(self, fmt: str, *args) -> None:
            # One tidy line per request instead of the default noise.
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--policy", type=Path, default=Path("policy.pt"))
    parser.add_argument("--lat-range", type=float, nargs=2, default=(-90.0, 90.0))
    parser.add_argument("--lon-range", type=float, nargs=2, default=(-180.0, 180.0))
    parser.add_argument("--alt-range", type=float, nargs=2, default=(0.0, 15000.0))
    parser.add_argument("--defences", type=int, default=5)
    parser.add_argument("--defence-spread", type=float, default=400.0)
    parser.add_argument("--blues", type=int, default=2)
    parser.add_argument("--blue-number", type=int, default=1)
    parser.add_argument("--blue-callsign", default="")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = MissionState(args)

    if state.model is None:
        print(f"  warning: no policy at {args.policy} — the Plan button will report this")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"  serving on {url}  (ctrl-c to stop)")

    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()