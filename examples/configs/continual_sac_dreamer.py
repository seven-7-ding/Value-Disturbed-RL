"""Continual SAC config with DreamerV3 size1m-aligned networks.

Network spec matching dreamerv3/configs.yaml (size1m):
    hidden_dims = (64, 64, 64)      # 3 layers × 64 units
    activation  = SiLU              (act: silu)
    norm        = none              (norm: none)
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ---------- SAC hyperparameters ----------
    config.actor_lr  = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr   = 3e-4

    # DreamerV3 size1m: 3 layers × 64 units
    config.hidden_dims      = (64, 64, 64)
    # Shorthand size preset (overrides hidden_dims when set).
    # Options: 'size1m', 'size12m', 'size25m', 'size50m', 'size100m', 'size200m', 'size400m'
    config.model_size       = 'size1m'
    config.discount         = 0.99
    config.tau              = 0.005
    config.init_temperature = 1.0
    config.target_entropy   = None
    config.backup_entropy   = True
    config.jax_mem_fraction = 0.4

    # ---------- ReDo (mirrors dreamerv3/configs.yaml 'redo' block) ----------
    redo = ml_collections.ConfigDict()
    redo.redo_enabled      = True
    redo.grad_redo_enabled = True
    redo.tau               = 0.05
    redo.mode              = 'threshold'
    # Steps between successive analysis+reset passes.
    # 10000 env-steps matches DreamerV3 default (frequency: 10000).
    redo.frequency         = 10000
    redo.log_item          = 'log+erank+srank'
    redo.skip_last_layer   = False
    redo.rank_threshold    = 0.99
    redo.reset_start       = 0
    redo.reset_end         = 0
    config.redo = redo

    return config
