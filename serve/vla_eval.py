"""vla_eval.py <task> <seed> <out.jsonl>: ONE pure-VLA evaluation episode under the official protocol (native reset, pretrain
object/kitchen split, official horizon from t50_horizons.json, no harness/VLM). Appends one jsonl line
{task, seed, h, success, within, steps, stop, wall_s, lang}; skips (task, seed) pairs already present without error."""
import os, sys, json, time
sys.path.insert(0, os.path.expanduser("~/ailab-via-demo-r9s")); sys.path.insert(0, "/tmp"); os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("VIA_WP_EARLY_POS", "0.004"); os.environ.setdefault("VIA_WP_EARLY_WINDOW", "6")
os.environ.setdefault("VIA_NATIVE_RESET", "1"); os.environ.setdefault("VIA_ROBOCASA_SPLIT", "pretrain")
TASK, SEED, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            d = json.loads(l)
            if d.get("task") == TASK and d.get("seed") == SEED and "error" not in d: print("exists", TASK, SEED); sys.exit(0)
        except Exception: pass
from via.sim_env import SimEnv
H = json.load(open("/data/weizhijie/via-demo/t50_horizons.json"))[TASK]
BRIDGES = os.environ["VIA_VLA_BRIDGE_URLS"].split(",")
t0 = time.time(); info = dict(task=TASK, seed=SEED, h=H, arm="VLA")
try:
    env = SimEnv(f"robocasa/{TASK}", on_screen_render=False, verbose=False, record_sim_state=False, controller_mode="pi", seed=SEED); env.reset()
    lang = str(env.language_instruction); bridge = BRIDGES[(hash(TASK) + SEED) % len(BRIDGES)]
    if os.environ.get("VIA_CAM_SCHED") == "1":  # pure-VLA eval: camera observables are unused (the policy renders its own 3 views) -> disable them;
        k = env.env                                  # and render the policy views only on the 4 steps per 16-step chunk that feed the policy (t, t-2, t-4, t-6)
        for n in [n for n in k.observation_names if n.endswith("_image") or n.endswith("_depth")]: k.modify_observable(n, "enabled", False)
        _orig = env._vla_images; _ctr = [0]; _last = [None]; _R = max(8, int(os.environ.get("VIA_VLA_REPLAN", "16") or 16))
        _need = {0, (_R - 2) % _R, (_R - 4) % _R, (_R - 6) % _R}   # pushes feeding the policy: call step t and t-2, t-4, t-6 (history interval 2)
        def _sched_images():
            _ctr[0] += 1
            if _last[0] is None or ((_ctr[0] - 1) % _R) in _need: _last[0] = _orig()
            return _last[0]
        env._vla_images = _sched_images; info["cam_sched"] = 1
    if os.environ.get("VIA_CAM_STRIDE2") == "1":  # render cameras every 2nd control step only (the policy uses frames t, t-2, t-4, t-6 at t % 16 == 0)
        k = env.env; names = [n for n in getattr(k, "observation_names", list(getattr(k, "_observables", {}).keys())) if n.endswith("_image") or n.endswith("_depth")]
        for n in names: k.modify_observable(n, "sampling_rate", float(k.control_freq) / 2)
        info["cam_stride2"] = len(names)
    r = env.run_vla_segment(lang, H, bridge, replan_steps=16, timeout_s=3600.0)
    success = bool(env.success or env.env._check_success())
    steps = r.get("steps", r.get("t", r.get("used")))
    info.update(success=success, within=success, steps=steps, stop=r.get("stop"), lang=lang, bridge=bridge.split("//")[-1].split("/")[0])
except Exception as ex:
    info.update(success=False, within=False, error=f"{type(ex).__name__}: {ex}"[:200])
info["wall_s"] = round(time.time() - t0, 1)
with open(OUT, "a") as f: f.write(json.dumps(info) + "\n")
print(json.dumps(info))
