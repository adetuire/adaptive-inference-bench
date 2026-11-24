# Adaptive Inference Bench

Local inference benchmark for the ELA classifier in [Ire Adetu's notebook](https://github.com/adetuire/deepfake_detection). Supply its full trained cnn_ela_model_weights.h5 and a directory of 256×256 images. Use a trusted checkpoint. The API preserves the notebook's quality-95 JPEG difference, brightness scaling and unnormalized pixel values. The score corresponds to label 1 (fake).

Install requirements in a Python virtual environment. Start the baseline:

    MODEL_PATH=/path/to/cnn_ela_model_weights.h5 BATCH_SIZE=1 uvicorn service:app

In a second terminal run:

    python bench.py /path/to/images > baseline.jsonl

Restart the server with BATCH_SIZE=8 and WAIT_MS=5, then repeat with the same images and save batched.jsonl. The runner sends 160 requests per concurrency level, cycling through the same sorted images. Report errors, throughput and p95 together. The client uses closed-loop concurrency, so the achieved request rate depends on the server. Batch model time includes array assembly and worker dispatch. Compare predictions with model(prepare(image)[None], training=False) before interpreting performance. The preprocessing microbenchmark runs independently of the checkpoint and measures only the image transform.

## Original checkpoint benchmark

One local macOS ARM64 CPU run on the trained ELA checkpoint, 160 requests per configuration, no HTTP failures. Used the last 32 alphabetically sorted JPEG paths from each class in full_dataset.zip, interleaving real and fake. This sample measures serving performance only. The baseline runs first. Results vary with hardware and load.

| Concurrent clients | Single requests/s | Batched requests/s | Single p95 ms | Batched p95 ms |
|---|---:|---:|---:|---:|
| 1 | 68.7 | 45.8 | 16.0 | 27.1 |
| 4 | 70.1 | 123.1 | 69.0 | 34.1 |
| 16 | 71.1 | 141.9 | 291.1 | 224.8 |

The five-millisecond batch window costs latency with one client. At 16 clients, batching amortized inference work and reduced queue time. See baseline.jsonl and batched.jsonl for measurements and benchmark-environment.txt for versions.
