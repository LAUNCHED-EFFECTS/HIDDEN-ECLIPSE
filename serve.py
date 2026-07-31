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

from globe import build_globe, render_html, route_traces_json
from plan import plan_mission
from ppo import load_policy
from world import World, generate_world


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
            a.defenses,
            a.defense_spread,
        )

    def new_scenario(self) -> None:
        with self.lock:
            self.world = self._generate()

    def page(self) -> str:
        with self.lock:
            world = self.world
        fig = build_globe(world.hostile, world.friendly, world.defenses)
        return render_html(fig, interactive=True)

    def plan(self) -> dict:
        if self.model is None:
            return {"error": f"no policy at {self.args.policy} — run train.py first"}

        with self.lock:
            scenario = self.world.to_scenario()

        mission = plan_mission(scenario, self.model, self.norm)
        payload = {
            "summary": mission.summary(),
            "succeeded": mission.succeeded,
            "traces": route_traces_json(mission),
        }
        if mission.threatening_site:
            gap = mission.closest_margin_km
            edge = "inside" if gap < 0 else "clear of"
            payload["summary"] += (
                f" · closest approach {abs(gap):,.1f} km {edge} {mission.threatening_site}"
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
    parser.add_argument("--defenses", type=int, default=5)
    parser.add_argument("--defense-spread", type=float, default=400.0)
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