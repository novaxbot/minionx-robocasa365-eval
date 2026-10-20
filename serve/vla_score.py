"""vla_score.py <run.jsonl> [ref.jsonl]: pure-VLA score = mean over tasks of per-task success rate (official protocol), with the
three target buckets; with ref, paired on common (task, seed) pairs; also prints the error count."""
import json, sys, collections
T = json.load(open("/data/weizhijie/via-demo/all_tasks.json")); bucket = {t: v["target"] for t, v in T.items() if v.get("target")}
def load(f):
    d = {}
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        d[(r["task"], r["seed"])] = r
    return d
run = load(sys.argv[1]); ref = load(sys.argv[2]) if len(sys.argv) > 2 else None
keys = sorted(run) if ref is None else sorted(set(run) & set(ref))
def score(d, keys):
    per = collections.defaultdict(list)
    for k in keys: per[k[0]].append(bool(d[k].get("within", d[k].get("success"))))
    tm = {t: sum(v) / len(v) for t, v in per.items()}
    agg = collections.defaultdict(list)
    for t, m in tm.items(): agg[bucket.get(t, "?")].append(m)
    return 100 * sum(tm.values()) / max(1, len(tm)), {b: round(100 * sum(v) / len(v), 1) for b, v in sorted(agg.items())}, len(tm)
s, b, nt = score(run, keys)
err = sum(1 for k in keys if "error" in run[k])
print(f"{sys.argv[1]}: episodes {len(keys)} tasks {nt} errors {err}  task-mean {s:.1f}  {b}")
if ref:
    s2, b2, _ = score(ref, keys); print(f"{sys.argv[2]}: paired task-mean {s2:.1f}  {b2}   delta {s - s2:+.1f}")
    per = collections.defaultdict(lambda: [0, 0, 0])
    for k in keys: per[k[0]][0] += bool(ref[k].get("within")); per[k[0]][1] += bool(run[k].get("within")); per[k[0]][2] += 1
    worst = sorted(per.items(), key=lambda kv: kv[1][1] - kv[1][0])
    print("largest drops:", [(t, f"{a}->{b}/{n}") for t, (a, b, n) in worst[:6]])
    print("largest gains:", [(t, f"{a}->{b}/{n}") for t, (a, b, n) in worst[::-1][:6]])
