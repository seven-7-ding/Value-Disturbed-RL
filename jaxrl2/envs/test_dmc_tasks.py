from dm_control import suite
import numpy as np

for domain, task in suite.ALL_TASKS:
    try:
        env = suite.load(domain, task)
        ts = env.reset()
        obs_dim = sum(np.asarray(v).flatten().shape[0] for v in ts.observation.values())
        act_dim = env.action_spec().shape[0]
        print(f"'{domain}_{task}': obs={obs_dim}, act={act_dim},")
    except Exception as e:
        print(f"# {domain}_{task}: {e}")
