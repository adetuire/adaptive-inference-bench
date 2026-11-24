import json, platform, time
from io import BytesIO
import numpy as np
from PIL import Image
from ela import prepare

rng = np.random.default_rng(42)
inputs = []
for _ in range(32):
    stream = BytesIO()
    Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8)).save(stream,format="JPEG",quality=90)
    inputs.append(stream.getvalue())
for image in inputs:
    prepare(image)
times = []
for image in inputs * 10:
    start = time.perf_counter()
    prepare(image)
    times.append(time.perf_counter()-start)
print(json.dumps(dict(stage="JPEG error-level preprocessing only", machine=platform.machine(),
                      samples=len(times), unique_synthetic_images=len(inputs),
                      images_per_second=len(times)/sum(times), p95_ms=1000*np.percentile(times,95)),indent=2))
