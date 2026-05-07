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

    # ---------- Optimizer (mirrors dreamerv3/configs.yaml opt block) ----------
    opt = ml_collections.ConfigDict()
    opt.agc      = 0.3       # adaptive gradient clipping ratio
    opt.eps      = 1e-20     # RMS denominator epsilon
    opt.beta1    = 0.9       # momentum decay
    opt.beta2    = 0.999     # RMS decay
    opt.momentum = True      # use first-moment accumulation
    opt.nesterov = False     # Nesterov momentum
    opt.wd       = 0.0       # weight decay (0 = disabled)
    opt.wdregex  = '/kernel$'  # regex selecting params for weight decay
    opt.schedule = 'const'   # lr schedule: 'const' | 'linear' | 'cosine'
    opt.warmup   = 1000      # linear warmup steps
    opt.anneal   = 0         # total steps for non-const schedule
    config.opt = opt

    return config
