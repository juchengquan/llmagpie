from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uuid
import time
import asyncio

app = FastAPI()

@app.post("/")
@app.get("/")
async def health():
    return "OK"

@app.post("/generate")
@app.get("/generate")
async def api_generate():
    dummy_res = {
        "generated_text": "What is the meaning of stonehenge?",
        "elapsed_time": 0.1,
        "trace_id": uuid.uuid4().hex
    }
    return dummy_res
    
def shell():
    generated_chunks = "What is the meaning of stonehenge?".split(" ")
    
    for chunk in generated_chunks:
        dummy_res = {
            "generated_text": chunk,
            "elapsed_time": 0.1,
            "trace_id": uuid.uuid4().hex
        }
        # await asyncio.sleep(0.5)
        time.sleep(0.25)

        # yield dummy_res
        yield chunk + " "
    
@app.post("/stream")
@app.get("/stream")
async def api_generate_stream():
    
    
    # return StreamingResponse(dummy_res)
    return StreamingResponse(shell())
