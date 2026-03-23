"""End-to-end smoke test for evaluation pipeline."""
import sys, os, gc, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pathlib import Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from evaluation.evaluate import extract_obs
from models.action_head_manager import ActionHeadManager

# Load model
ckpt_dir = '/tmp/test_ckpt_diffusion'
l1_classes = []
for entry in sorted(os.listdir(ckpt_dir)):
    ckpt_path = os.path.join(ckpt_dir, entry, 'diffusion', 'best.pt')
    if os.path.isfile(ckpt_path):
        meta = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        l1_classes.append(meta.get('l1_class', entry))
        del meta
print('L1 classes:', l1_classes)

manager = ActionHeadManager(head_type='diffusion', l1_classes=l1_classes, T_obs=2, T_pred=16)
manager.load_all(ckpt_dir, use_ema=True)
manager.eval()

# Load action stats
action_stats = {}
for cls_name in l1_classes:
    key = cls_name.replace('.', '_')
    sp = os.path.join(ckpt_dir, key, 'diffusion', 'action_stats.json')
    if os.path.isfile(sp):
        with open(sp) as f:
            action_stats[cls_name] = json.load(f)

# Setup env
task_suite = benchmark.get_benchmark_dict()['libero_90']()
bddl_root = Path(get_libero_path('bddl_files'))
task = task_suite.get_task(0)
bddl_file = str(bddl_root / task.problem_folder / task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)

init_states = task_suite.get_task_init_states(0)
env.reset()
raw_obs = env.set_init_state(init_states[0])
for _ in range(5):
    raw_obs, _, _, _ = env.step(np.zeros(7))

obs = extract_obs(raw_obs)
av_hist = [obs['agentview'].copy(), obs['agentview'].copy()]
wr_hist = [obs['wrist'].copy(), obs['wrist'].copy()]
pr_hist = [obs['proprio'].copy(), obs['proprio'].copy()]

# Run 3 action chunks
for chunk_i in range(3):
    av = np.stack(av_hist[-2:])
    wr = np.stack(wr_hist[-2:])
    pr = np.stack(pr_hist[-2:])

    av_t = torch.from_numpy(av.transpose(0, 3, 1, 2).astype(np.float32) / 255.0).unsqueeze(0)
    wr_t = torch.from_numpy(wr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0).unsqueeze(0)
    pr_t = torch.from_numpy(pr.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        actions = manager.predict(av_t, wr_t, pr_t, 'push-12-1')
    chunk = actions[0].cpu().numpy()

    # Denormalize
    stats = action_stats.get('push-12-1')
    if stats:
        a_min = np.array(stats['min'], dtype=np.float32)
        a_max = np.array(stats['max'], dtype=np.float32)
        rng = np.maximum(a_max - a_min, 1e-6)
        chunk = ((chunk + 1) / 2 * rng + a_min).astype(np.float32)

    for act in chunk:
        raw_obs, rew, done, info = env.step(act)
        obs = extract_obs(raw_obs)
        av_hist.append(obs['agentview'])
        wr_hist.append(obs['wrist'])
        pr_hist.append(obs['proprio'])
        if done:
            break

    print(f'Chunk {chunk_i}: done={done}, '
          f'actions range=[{chunk.min():.4f}, {chunk.max():.4f}]')
    if done:
        break

env.close()
del env
gc.collect()
print('END-TO-END TEST PASSED')
