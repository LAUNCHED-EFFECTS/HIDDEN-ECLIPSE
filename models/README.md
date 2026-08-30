# models

`policy.pt` is the trained PPO checkpoint the planner loads by default — a
state dict plus the observation statistics the network was normalised with, so
a checkpoint is self-contained.

    python3 bin/train.py --output models/policy.pt      # write a new one
    python3 bin/evaluate.py --policy models/policy.pt   # score it