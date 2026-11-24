"""Closed-loop local HTTP benchmark. Run separately for each server configuration."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
import numpy as np

def run(paths, clients, count):
    images = [p.read_bytes() for p in paths]
    def call(index):
        start = time.perf_counter()
        try:
            with urlopen(Request("http://127.0.0.1:8000/predict", data=images[index % len(images)]), timeout=60) as response:
                result = json.load(response)
            return (1000*(time.perf_counter()-start), result["queue_ms"], result["batch_model_ms"])
        except (URLError, TimeoutError):
            return None
    with ThreadPoolExecutor(clients) as pool:
        start = time.perf_counter()
        rows = [r for r in pool.map(call, range(count)) if r is not None]
        seconds = time.perf_counter()-start
    return dict(clients=clients, requests=count, failures=count-len(rows), successful_per_second=len(rows)/seconds,
                p95_ms=dict(zip(("end_to_end","queue","batch_model"),np.percentile(rows,95,axis=0))) if rows else {})

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images",type=Path)
    args=parser.parse_args()
    files=sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"})[:64]
    if not files:
        parser.error("No benchmark images")
    for concurrency in (1,4,16):
        print(json.dumps(run(files,concurrency,160)),flush=True)
