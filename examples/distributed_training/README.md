# Distributed Training

## Single Node Multiple GPUs

```bash
accelerate launch --config_file examples/distributed_training/single_node.json examples/model/train_ranker.py
```

## Multiple Nodes

Launch the following command on each node:

```bash
# rank 0
accelerate launch --config_file examples/distributed_training/multi_nodes_rank0.json examples/model/train_ranker.py

# rank 1
accelerate launch --config_file examples/distributed_training/multi_nodes_rank1.json examples/model/train_ranker.py
```

