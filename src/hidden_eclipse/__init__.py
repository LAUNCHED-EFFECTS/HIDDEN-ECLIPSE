"""Hidden Eclipse: a mission-planning sandbox over great-circle geometry.

The package holds the pieces the entry points in `bin/` share — geospatial
primitives, the air-defence laydown, the environment the policy is trained
against, the PPO implementation, the planner, and the globe renderer.
"""

from __future__ import annotations

__all__ = ["defences", "env", "geo", "globe", "paths", "plan", "ppo", "world"]
