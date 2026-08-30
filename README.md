# Hidden Eclipse

A mission-planning sandbox. A random target, a random strike package, and a
random air-defence laydown on a globe you can drag things around on — with a
PPO policy that routes the package onto the target and tells you how it went.

## Layout

    bin/                  entry points — the only interface
      globe.py            generate a scenario, write (and open) the globe
      serve.py            the same globe with working mission-control buttons
      train.py            train the PPO planner
      evaluate.py         score a checkpoint against baselines
    src/hidden_eclipse/   the library everything above shares
      geo.py              positions and great-circle math
      defences.py         air-defence sites and threat checks
      env.py              the episode the policy is trained against
      world.py            scenario construction, callsigns, package editing
      ppo.py              network, rollout buffer, trainer
      plan.py             policy in, mission plan out
      globe.py            plotly rendering and the page's JavaScript
      paths.py            where the artifacts live
    tests/                pytest suite
    demo/                 a rendered globe to look at without running anything
    models/               trained checkpoints (`policy.pt` ships with the repo)

## Getting started

The virtual environment lives in the repository, at `hidden-eclipse/`:

    python3 -m venv hidden-eclipse
    source hidden-eclipse/bin/activate
    pip install -e ".[dev]"

That installs the package in editable mode, so `bin/` scripts and tests import
`hidden_eclipse` without any path juggling and edits to `src/` take effect
immediately. `requirements.txt` holds the exact pinned versions if you need to
reproduce the environment rather than resolve a fresh one.

## Running it

    python3 bin/globe.py --seed 1337          # a repeatable scenario
    python3 bin/globe.py --plan               # route the package with the policy
    python3 bin/serve.py                      # http://127.0.0.1:8000, buttons live
    python3 bin/evaluate.py --episodes 400    # score the checkpoint
    python3 bin/train.py --steps 100000       # train a new one

Every script takes `--help`. Defaults resolve against the repository root, so
they work from any directory.

## Tests

    pytest