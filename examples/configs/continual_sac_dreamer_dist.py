"""Continual SAC-Dreamer **Distributional** config.

Uses SACDreamerDistLearner:
  - Critic: TwoHot distributional (255 bins, symexp-spaced ±20)
  - Critic loss: cross-entropy vs two-hot target  (mirrors DreamerV3 value loss)
  - Actor Q: _twohot_pred() expected value
  - Networks: same SiLU+RMSNorm MLP as SACDreamerLearner (size1m default)
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
    config.model_size       = 'size1m'
    # TwoHot bins (mirrors DreamerV3 default: 255 symexp-spaced bins in [-20, 20])
    config.num_bins         = 255
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
    redo.frequency         = 10000
    redo.log_item          = 'log+erank+srank'
    redo.skip_last_layer   = False
    redo.rank_threshold    = 0.99
    redo.reset_start       = 0
    redo.reset_end         = 0
    config.redo = redo

    return config
