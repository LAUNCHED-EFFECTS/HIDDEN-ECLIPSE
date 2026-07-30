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

from geo import great_circle_km, initial_bearing_deg, random_position
from globe import build_globe


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
        "--no-open",
        action="store_true",
        help="write the file without opening a browser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    hostile = random_position(rng, tuple(args.lat_range), tuple(args.lon_range))
    friendly = random_position(rng, tuple(args.lat_range), tuple(args.lon_range))

    separation = great_circle_km(hostile, friendly)
    bearing = initial_bearing_deg(friendly, hostile)

    print(f"  RED  (hostile)  {hostile}")
    print(f"  BLUE (friendly) {friendly}")
    print(f"  separation      {separation:,.1f} km")
    print(f"  bearing to RED  {bearing:.1f}° true")
    if args.seed is not None:
        print(f"  seed            {args.seed}")

    output = args.output.resolve()
    build_globe(hostile, friendly).write_html(str(output), include_plotlyjs=True)
    print(f"\n  globe written to {output}")

    if not args.no_open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
