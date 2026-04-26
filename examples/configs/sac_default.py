import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.tau = 0.005
    config.init_temperature = 1.0
    config.target_entropy = None
    config.backup_entropy = True
    config.jax_mem_fraction = 0.4  # JAX 可使用的最大 GPU 显存比例 (0.0~1.0)

    return config
