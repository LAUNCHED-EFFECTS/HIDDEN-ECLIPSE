#!/usr/bin/env python3
"""Generate a random hostile (red) and friendly (blue) position and plot them.

    python3 bin/globe.py                  # new random pair, opens the globe
    python3 bin/globe.py --seed 1337      # reproducible pair
    python3 bin/globe.py --no-open        # just write the HTML
"""

from __future__ import annotations

import argparse
import random
import webbrowser
from pathlib import Path

from hidden_eclipse.defences import engaging
from hidden_eclipse.geo import (
    elevation_angle_deg,
    great_circle_km,
    initial_bearing_deg,
    slant_range_km,
)
from hidden_eclipse.globe import build_globe, write_globe
from hidden_eclipse.paths import DEFAULT_GLOBE, DEFAULT_POLICY
from hidden_eclipse.world import generate_world
from hidden_eclipse.plan import load_and_plan


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
        default=DEFAULT_GLOBE,
        help="where to write the interactive globe (default: demo/globe.html)",
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
        "--blues",
        type=int,
        default=2,
        metavar="N",
        help="number of friendly assets in the package (default: 2)",
    )
    parser.add_argument(
        "--blue-number",
        type=int,
        default=1,
        metavar="N",
        help="asset number shown as 'BLUE N' (default: 1)",
    )
    parser.add_argument(
        "--blue-callsign",
        default="",
        metavar="NAME",
        help="optional callsign, rendered as 'BLUE N (NAME)'",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="route BLUE onto RED with the trained PPO policy",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="policy checkpoint used by --plan (default: models/policy.pt)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="write the file without opening a browser",
    )
    return parser.parse_args()


def build_plan(args, world):
    """Route the package onto RED, and print the briefing."""
    if not args.policy.exists():
        print(f"\n  no policy at {args.policy} — run `python3 bin/train.py` first")
        return None

    mission = load_and_plan(
        args.policy, world.to_scenario(), labels=world.friendly_labels
    )
    print(f"\n  mission plan: {mission.summary()}")

    for asset in mission.assets:
        print(f"\n  {asset.summary()}  ({asset.route_km:,.0f} km, "
              f"{len(asset.waypoints)} waypoints)")
        if asset.threatening_site:
            gap = asset.closest_margin_km
            edge = "inside" if gap < 0 else "clear of"
            print(f"    closest approach {abs(gap):,.1f} km {edge} "
                  f"{asset.threatening_site}")

        print(f"    {'wp':>3}  {'position':<26} {'alt':>9}  {'hdg':>5}  {'T+':>7}")
        for i, wp in enumerate(asset.waypoints, 1):
            print(
                f"    {i:>3}  {wp.position.coords:<26} {wp.position.alt_m:>7,.0f} m"
                f"  {wp.heading:>4.0f}°  {wp.elapsed_min:>5,.0f} min"
            )
    return mission


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    world = generate_world(
        rng,
        tuple(args.lat_range),
        tuple(args.lon_range),
        tuple(args.alt_range),
        args.defences,
        args.defence_spread,
        args.blue_number,
        args.blue_callsign,
        args.blues,
    )
    hostile, sites = world.hostile, world.defences
    friendlies, labels = world.friendlies, world.friendly_labels

    # Width follows the labels, which grow with the callsign.
    width = max([len(l) for l in labels] + [len("RED")]) + 2
    print(f"  {'RED':<{width}} {hostile}")
    for label, position in zip(labels, friendlies):
        print(f"  {label:<{width}} {position}")

    lead = friendlies[0]
    print(f"\n  lead asset {labels[0]} to RED")
    print(f"    ground range     {great_circle_km(hostile, lead):,.1f} km")
    print(f"    slant range      {slant_range_km(hostile, lead):,.1f} km")
    print(f"    bearing          {initial_bearing_deg(lead, hostile):.1f}° true")
    print(f"    elevation        {elevation_angle_deg(lead, hostile):+.2f}°")
    if args.seed is not None:
        print(f"  seed             {args.seed}")

    if sites:
        print(f"\n  enemy air defence ({len(sites)} sites within "
              f"{args.defence_spread:,.0f} km of RED)")
        for site in sites:
            print(f"    {site}")

        for label, position in zip(labels, friendlies):
            threats = engaging(sites, position)
            if threats:
                names = ", ".join(s.designator for s in threats)
                print(f"\n  {label} is inside the envelope of: {names}")
            else:
                print(f"\n  {label} is outside every engagement envelope")

    mission = None
    if args.plan:
        mission = build_plan(args, world)

    output = args.output.resolve()
    write_globe(
        build_globe(hostile, friendlies, sites, mission, friendly_names=labels), output
    )
    print(f"\n  globe written to {output}")

    if not args.no_open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
