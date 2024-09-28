net_params = {
    "L": 4,
    "hidden_dim": 70,
    "out_dim": 60,
    "residual": True,
    "edge_feat": True,
    "readout": "sum",
    "graph_norm": True,
    "batch_norm": True,
    "towers": 5,
    "divide_input_first": True,
    "divide_input_last": True,
    "edge_dim": 50,
    "pretrans_layers" : 1,
    "posttrans_layers" : 1,
    "num_atom_type": 28,
    "num_bond_type": 4
}

optimization_params = {
    "epochs": 1000,
    "batch_size": 128,
    "init_lr": 1e-3,
    "min_lr": 1e-5,
    "weight_decay": 3e-6,
    "lr_reduce_factor": 0.5,
    "lr_schedule_patience": 20,
    "early_stopping": 50
}