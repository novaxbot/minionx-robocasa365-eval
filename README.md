# minionx — RoboCasa365 evaluation materials

This private repository accompanies the RoboCasa365 leaderboard submission **minionx** (submitter: novaxbot,
leaderboard PR robocasa-benchmark/leaderboard#35). It contains the evaluation code used for the reported numbers,
the per-episode results, and the exact list of official seeds that were run.

## Model

- Checkpoint (private Hugging Face repo, access granted to the RoboCasa team): `novaxbot/minionx-robocasa365`
- minionx is Xiaomi-Robotics-1-RoboCasa365 fine-tuned with multi-teacher on-policy distillation; all weights were trained.
- The checkpoint is a standard Hugging Face folder (3 safetensors shards, config, tokenizer/processor files and the
  `modeling_mibot.py` / `processing_mibot.py` / `configuration_mibot.py` remote code). It loads exactly like the Xiaomi
  release: `AutoModel.from_pretrained(<ckpt>, trust_remote_code=True)` and `AutoProcessor.from_pretrained(<ckpt>, trust_remote_code=True)`.
  Inference code is unchanged from Xiaomi-Robotics-1.

## Recommended way to evaluate (official pipeline)

Use the official Xiaomi-Robotics-1 RoboCasa365 evaluation code
(https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/tree/main/eval_robocasa365, commit 0dd7aef8dc87296246aae812a1f59ccb708e5546)
and point its model server at the minionx checkpoint instead of `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`.
Settings are the Xiaomi defaults: RoboCasa 1.0.1, `split="pretrain"`, `--replan-steps 16 --obs-history 4 --obs-interval 2 --crop-ratio 0.95`,
task horizons from `meta/t50_horizons.json`, seed `7 + task_index * 50 + episode` for 50 episodes per task.

## What was run for the reported numbers

The reported numbers were produced with our in-house evaluation harness (same protocol: native env reset, `split=pretrain`,
official seeds, official horizons, 16-step action chunks, 4-frame observation history with interval 2, crop ratio 0.95):

- `serve/server_h100.py` — model worker (plain PyTorch, loads the checkpoint with `trust_remote_code`, returns 16-step action chunks)
- `serve/gateway.py` — request gateway over the workers; `serve/vla_bridge.py` — HTTP bridge that builds the model inputs
  (3 camera views x 4 history frames, 14-D state, instruction) and returns the action chunk; `serve/h100_serve.sh` — launcher
- `serve/vla_eval.py` — one evaluation episode (task, seed) under the official protocol; `serve/vla_launch_cpu.sh` — lane launcher
  (CPU rendering); `serve/vla_score.py` — scoring. These two files depend on our internal RoboCasa simulation client (`via.sim_env`),
  which is not included; they are provided for transparency of the protocol, not as a standalone runner.
- `plans/official_50x50.txt` — the 2,500 (task, seed) pairs (official seed rule), `meta/all_tasks.json` — task → split mapping.

## Results (RoboCasa 1.0.1, split=pretrain, 50 tasks x 50 episodes, 2026-10-19)

| split | successes / episodes | success rate |
|---|---|---|
| atomic_seen (18 tasks) | 728 / 900 | 80.9 |
| composite_seen (16 tasks) | 464 / 800 | 58.0 |
| composite_unseen (16 tasks) | 361 / 797 | 45.3 |
| task-weighted overall | 1553 / 2497 | 62.2 |

Three episodes were not run (WaffleReheat, seeds 2373, 2376 and 2378) and are excluded from the denominator.
Per-episode records: `results/official_t4_a.jsonl` and `results/official_t4_b.jsonl` (one JSON per episode:
task, seed, horizon `h`, `success`, `steps`, `stop`, instruction `lang`, wall time).
