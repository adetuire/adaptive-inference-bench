"""Launch with MODEL_PATH pointing to a trusted local full Keras ELA checkpoint."""
import asyncio
import os
import time
from contextlib import asynccontextmanager, suppress
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from ela import prepare


async def serve(queue, model, limit, window):
    while True:
        batch = [await queue.get()]
        deadline = time.perf_counter() + window
        while len(batch) < limit:
            try:
                batch.append(await asyncio.wait_for(queue.get(), deadline - time.perf_counter()))
            except asyncio.TimeoutError:
                break
        started = time.perf_counter()
        try:
            images = np.stack([item[0] for item in batch])
            scores = await asyncio.to_thread(lambda: model(images, training=False).numpy().reshape(-1))
            elapsed = 1000 * (time.perf_counter() - started)
            for (_, submitted, result), score in zip(batch, scores):
                if not result.done():
                    result.set_result(dict(fake_score=float(score), batch_size=len(batch),
                                           queue_ms=1000*(started-submitted), batch_model_ms=elapsed))
        except Exception as exc:
            for _, _, result in batch:
                if not result.done():
                    result.set_exception(exc)


@asynccontextmanager
async def lifespan(app):
    from tensorflow import keras
    model = keras.models.load_model(os.environ["MODEL_PATH"], compile=False)
    if tuple(model.input_shape[1:]) != (256, 256, 3) or model.output_shape[-1] != 1:
        raise ValueError("Expected the notebook's ELA sigmoid classifier")
    model(np.zeros((1,256,256,3), dtype=np.uint8), training=False)
    app.state.queue = asyncio.Queue(maxsize=128)
    task = asyncio.create_task(serve(app.state.queue, model, int(os.getenv("BATCH_SIZE","1")),
                                     float(os.getenv("WAIT_MS","5")) / 1000))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request):
    try:
        image = await asyncio.to_thread(prepare, await request.body())
    except (ValueError, OSError) as exc:
        raise HTTPException(400, "Expected a valid 256 by 256 image") from exc
    result = asyncio.get_running_loop().create_future()
    try:
        app.state.queue.put_nowait((image, time.perf_counter(), result))
    except asyncio.QueueFull:
        raise HTTPException(503, "Queue full")
    return await result
