#!/usr/bin/env python3
"""AI-reader stage 2: send prepared jobs (work/ai/jobs) to Gemini exactly as the production readers do
(google-genai client with a 300 s timeout; contents = [image part(s)..., prompt]; GenerateContentConfig with
response_mime_type application/json, the production response_schema, temperature 0; no thinking override).

The API key is read by the client from the environment (GEMINI_API_KEY); it is never printed or stored.
Every answered call is cached in work/ai/gemini/<key>.json with its usage metadata and appended to
work/ai/gemini_ledger.jsonl; failed calls are never cached. A hard budget guard stops issuing calls once the
conservative cost estimate of the calls made so far (this and earlier runs) plus one worst-case call would pass
--cap-usd.

Prices (US$ per million tokens), gemini-3.8-flash: list $0.75 input / $3.75 output (thinking billed as output),
as quoted by third-party price trackers (Oct 2026; not confirmed on Google's page). Conservative estimate used for
the guard: 2x list ($1.50 / $7.50), the rate reported to apply from Jan 2027.

  ~/orchid-py313/bin/python bench/ai/run_gemini.py --cap-usd 20 [--match <substring>] [--docs d1,d2] [--workers 8]
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
AI = BENCH / "work" / "ai"
CACHE = AI / "gemini"
LEDGER = AI / "gemini_ledger.jsonl"
MODEL = "gemini-3.8-flash"
LIST_IN, LIST_OUT = 0.75, 3.75
CONS_IN, CONS_OUT = 1.50, 7.50
LOCK = threading.Lock()


def cost(u, pin, pout):
    """Return the USD cost of one Gemini call from its token usage and per-million token prices."""
    if not u:
        return 0.0
    out = (u.get("candidates_token_count") or 0) + (u.get("thoughts_token_count") or 0)
    return ((u.get("prompt_token_count") or 0) * pin + out * pout) / 1e6


def spent():
    """Return the USD spend recorded in the ledger so far."""
    tot_list = tot_cons = 0.0
    if LEDGER.exists():
        for ln in LEDGER.read_text().splitlines():
            r = json.loads(ln)
            tot_list += cost(r.get("usage"), LIST_IN, LIST_OUT)
            tot_cons += cost(r.get("usage"), CONS_IN, CONS_OUT)
    return tot_list, tot_cons


_CLIENT = None


def client():
    """Return the shared Gemini API client, creating it on first use."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        from google.genai import types

        _CLIENT = genai.Client(http_options=types.HttpOptions(timeout=300_000))
    return _CLIENT


def call(job):
    """Send one AI job with its images to Gemini."""
    from google.genai import types

    parts = [types.Part.from_bytes(data=(BENCH / p).read_bytes(), mime_type="image/png") for p in job["images"]]
    t0 = time.time()
    last = None
    for attempt in range(4):
        try:
            r = client().models.generate_content(
                model=MODEL,
                contents=parts + [job["prompt"]],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=job["schema"], temperature=0
                ),
            )
            um = r.usage_metadata
            usage = (
                {
                    k: getattr(um, k, None)
                    for k in (
                        "prompt_token_count",
                        "cached_content_token_count",
                        "candidates_token_count",
                        "thoughts_token_count",
                        "total_token_count",
                    )
                }
                if um
                else None
            )
            try:
                result = json.loads(r.text)
            except Exception as exc:  # noqa: BLE001  - billed but unusable: recorded, not cached as an answer
                return {"error": f"unparseable JSON: {exc}"[:200], "usage": usage, "seconds": round(time.time() - t0, 1)}
            return {"result": result, "usage": usage, "seconds": round(time.time() - t0, 1)}
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            if any(s in last for s in ("RESOURCE_EXHAUSTED", "quota", "API key", "PERMISSION_DENIED")):
                break
            time.sleep(4 * (attempt + 1))
    return {"error": (last or "unknown")[:300], "usage": None, "seconds": round(time.time() - t0, 1)}


def main():
    """Run the budgeted Gemini jobs (command-line entry point)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-usd", type=float, required=True, help="conservative-estimate budget across all runs")
    ap.add_argument("--docs", default="")
    ap.add_argument("--renderings", default="")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--worst-call-usd", type=float, default=0.25)
    ap.add_argument("--jobs-dir", default=str(AI / "jobs"))
    a = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("no Gemini API key in the environment: stop")
    CACHE.mkdir(parents=True, exist_ok=True)
    docs = set(filter(None, a.docs.split(",")))
    rends = set(filter(None, a.renderings.split(",")))
    kinds = set(filter(None, a.kinds.split(",")))
    jobs = []
    for f in sorted(Path(a.jobs_dir).glob("*.json")):
        j = json.loads(f.read_text())
        if (docs and j["doc"] not in docs) or (rends and j["rendering"] not in rends) or (kinds and j["kind"] not in kinds):
            continue
        if (CACHE / f"{j['key']}.json").exists():
            continue
        jobs.append(j)
    print(f"{len(jobs)} uncached jobs; spent so far list ${spent()[0]:.4f} / conservative ${spent()[1]:.4f}", flush=True)
    stop = threading.Event()

    def work(j):
        with LOCK:
            if stop.is_set() or spent()[1] + a.worst_call_usd * a.workers > a.cap_usd:
                stop.set()
                return j["key"], "skipped (budget guard)"
        res = call(j)
        with LOCK:
            with LEDGER.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "key": j["key"],
                            "kind": j["kind"],
                            "doc": j["doc"],
                            "rendering": j["rendering"],
                            "model": MODEL,
                            "usage": res.get("usage"),
                            "seconds": res.get("seconds"),
                            "error": res.get("error"),
                        }
                    )
                    + "\n"
                )
            if "result" in res:
                (CACHE / f"{j['key']}.json").write_text(
                    json.dumps({"model": MODEL, "key": j["key"], **res}, ensure_ascii=False, indent=1)
                )
        return j["key"], res.get("error") or f"ok {cost(res.get('usage'), LIST_IN, LIST_OUT):.4f}"

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, s in ex.map(work, jobs):
            print(k, s, flush=True)
    L, C = spent()
    print(f"spent: list ${L:.4f}, conservative ${C:.4f}", flush=True)


if __name__ == "__main__":
    main()
