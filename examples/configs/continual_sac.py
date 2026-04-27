import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ---------- SAC hyperparameters ----------
    config.actor_lr  = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr   = 3e-4

    config.hidden_dims      = (256, 256)
    config.discount         = 0.99
    config.tau              = 0.005
    config.init_temperature = 1.0
    config.target_entropy   = None
    config.backup_entropy   = True
    config.jax_mem_fraction = 0.4

    # ---------- ReDo (mirrors dreamerv3/configs.yaml 'redo' block) ----------
    redo = ml_collections.ConfigDict()
    # Set True to enable activation-based FGReDo.
    redo.redo_enabled      = True
    # Set True to enable gradient-based FGGradientReDo.
    redo.grad_redo_enabled = True
    # Dormancy threshold (threshold mode) or fraction (percentage mode).
    redo.tau               = 0.05
    # 'threshold': dormant if normalised score <= tau
    # 'percentage': dormant if in the lowest tau-fraction
    redo.mode              = 'threshold'
    # Steps between successive analysis+reset passes.
    redo.frequency         = 640000
    # What to log/do each pass (substring-matched, combinable with '+'):
    #   'disabled' – skip all work
    #   'log'      – log dormancy metrics only
    #   'reset'    – log + reinitialise dormant weights
    #   'erank'    – also compute effective rank (SVD)
    #   'srank'    – also compute stable rank (SVD)
    # Examples: 'log', 'reset', 'log+erank', 'reset+erank+srank'
    redo.log_item          = 'log+erank+srank'
    # Skip the last executed layer (mirrors PyTorch FGReDo behaviour).
    redo.skip_last_layer   = False
    # Cumulative singular-value fraction for stable rank.
    redo.rank_threshold    = 0.99
    # Optional step-range window for resets (0 = no limit).
    redo.reset_start       = 0
    redo.reset_end         = 0
    config.redo = redo

    return config
