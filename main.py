"""Generate a random hostile (red) and friendly (blue) position and plot them.

    python3 main.py                  # new random pair, opens the globe
    python3 main.py --seed 1337      # reproducible pair
    python3 main.py --no-open        # just write the HTML
"""

from __future__ import annotations

import argparse
import random
import webbrowser
from pathlib import Path

from defences import engaging, random_defences
from env import Scenario
from geo import (
    elevation_angle_deg,
    great_circle_km,
    initial_bearing_deg,
    random_position,
    slant_range_km,
)
from globe import build_globe, write_globe
from plan import load_and_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the generator for a repeatable pair of positions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("globe.html"),
        help="where to write the interactive globe (default: globe.html)",
    )
    parser.add_argument(
        "--lat-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-90.0, 90.0),
        help="constrain sampling to a latitude band",
    )
    parser.add_argument(
        "--lon-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-180.0, 180.0),
        help="constrain sampling to a longitude band",
    )
    parser.add_argument(
        "--alt-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 15000.0),
        help="altitude band to sample, in metres MSL (default: 0 15000)",
    )
    parser.add_argument(
        "--defences",
        type=int,
        default=5,
        metavar="N",
        help="number of enemy air-defence sites around RED (default: 5, 0 for none)",
    )
    parser.add_argument(
        "--defence-spread",
        type=float,
        default=400.0,
        metavar="KM",
        help="radius around RED within which sites are scattered (default: 400)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="route BLUE onto RED with the trained PPO policy",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policy.pt"),
        help="policy checkpoint used by --plan (default: policy.pt)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="write the file without opening a browser",
    )
    return parser.parse_args()


def build_plan(args, hostile, friendly, sites):
    """Route BLUE onto RED with the trained policy, and print the briefing."""
    if not args.policy.exists():
        print(f"\n  no policy at {args.policy} — run `python3 train.py` first")
        return None

    scenario = Scenario(
        target=hostile,
        start=friendly,
        start_heading=initial_bearing_deg(friendly, hostile),
        defences=sites,
    )
    mission = load_and_plan(args.policy, scenario)

    print(f"\n  mission plan: {mission.summary()}")
    if mission.threatening_site:
        gap = mission.closest_margin_km
        edge = "inside" if gap < 0 else "clear of"
        print(f"  closest approach {abs(gap):,.1f} km {edge} {mission.threatening_site}")

    print(f"\n  {'wp':>3}  {'position':<26} {'alt':>9}  {'hdg':>5}  {'T+':>7}")
    for i, wp in enumerate(mission.waypoints, 1):
        print(
            f"  {i:>3}  {wp.position.coords:<26} {wp.position.alt_m:>7,.0f} m"
            f"  {wp.heading:>4.0f}°  {wp.elapsed_min:>5,.0f} min"
        )
    return mission


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ranges = (tuple(args.lat_range), tuple(args.lon_range), tuple(args.alt_range))
    hostile = random_position(rng, *ranges)
    friendly = random_position(rng, *ranges)

    print(f"  RED  (hostile)   {hostile}")
    print(f"  BLUE (friendly)  {friendly}")
    print(f"  ground range     {great_circle_km(hostile, friendly):,.1f} km")
    print(f"  slant range      {slant_range_km(hostile, friendly):,.1f} km")
    print(f"  bearing to RED   {initial_bearing_deg(friendly, hostile):.1f}° true")
    print(f"  elevation to RED {elevation_angle_deg(friendly, hostile):+.2f}°")
    if args.seed is not None:
        print(f"  seed             {args.seed}")

    sites = random_defences(hostile, rng, args.defences, args.defence_spread)
    if sites:
        print(f"\n  enemy air defence ({len(sites)} sites within "
              f"{args.defence_spread:,.0f} km of RED)")
        for site in sites:
            print(f"    {site}")

        threats = engaging(sites, friendly)
        if threats:
            names = ", ".join(s.designator for s in threats)
            print(f"\n  BLUE is inside the envelope of: {names}")
        else:
            print("\n  BLUE is outside every engagement envelope")

    mission = None
    if args.plan:
        mission = build_plan(args, hostile, friendly, sites)

    output = args.output.resolve()
    write_globe(build_globe(hostile, friendly, sites, mission), output)
    print(f"\n  globe written to {output}")

    if not args.no_open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
