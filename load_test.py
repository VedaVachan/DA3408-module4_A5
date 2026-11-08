"""
load_test.py -- generic closed-loop load generator for any model endpoint (REST / MLflow / Ray).
  python load_test.py --url http://localhost:8000/v1/models/churn:predict --format rest
  python load_test.py --url http://localhost:5001/invocations --format mlflow --concurrency 1 4 8
Reports p50/p95/p99 latency, throughput, errors and which replicas / model versions answered.
"""
import argparse, json, time, statistics, collections, concurrent.futures as cf
import pandas as pd, requests

def payload(rows, fmt, batch):
    recs = rows[:batch]
    if fmt == "rest":   return {"instances": recs}
    if fmt == "mlflow": return {"dataframe_records": recs}
    raise ValueError(fmt)

def one(sess, url, body):
    t0 = time.perf_counter()
    try:
        r = sess.post(url, json=body, timeout=30)
        ok = r.status_code == 200
        j = r.json() if ok else {}
        who = j.get("served_by", "n/a") if isinstance(j, dict) else "n/a"
        ver = j.get("model_version", "n/a") if isinstance(j, dict) else "n/a"
        if ver == "n/a" and isinstance(j, dict) and j.get("predictions") and isinstance(j["predictions"][0], dict):
            ver = j["predictions"][0].get("model_version", "n/a")      # MLflow pyfunc output
    except Exception:
        ok, who, ver = False, "n/a", "n/a"
    return (time.perf_counter() - t0) * 1000, ok, who, ver

def run(url, fmt, conc, duration, rows, batch):
    body = payload(rows, fmt, batch)
    lat, errs, who, ver = [], 0, collections.Counter(), collections.Counter()
    end = time.time() + duration
    def worker():
        s = requests.Session(); out = []
        while time.time() < end:
            out.append(one(s, url, body))
        return out
    with cf.ThreadPoolExecutor(conc) as ex:
        for res in [f.result() for f in [ex.submit(worker) for _ in range(conc)]]:
            for ms, ok, w, v in res:
                lat.append(ms); errs += (not ok); who[w] += 1; ver[v] += 1
    lat.sort(); n = len(lat); q = lambda p: lat[min(n-1, int(p*n))]
    return dict(concurrency=conc, requests=n, rps=round(n/duration, 1), p50_ms=round(q(.5), 1),
                p95_ms=round(q(.95), 1), p99_ms=round(q(.99), 1), errors=errs,
                replicas=len(who), versions=dict(ver))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--format", default="rest")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--duration", type=float, default=5); ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--data", default="data/sample_requests.csv"); ap.add_argument("--out")
    a = ap.parse_args()
    rows = pd.read_csv(a.data).to_dict(orient="records")
    results = [run(a.url, a.format, c, a.duration, rows, a.batch) for c in a.concurrency]
    print(pd.DataFrame(results).to_string(index=False))
    if a.out: json.dump(results, open(a.out, "w"), indent=2)
