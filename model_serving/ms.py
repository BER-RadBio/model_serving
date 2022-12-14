from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import logging


import uvicorn
from pydantic import BaseModel, Field
from energonai import launch_engine, QueueFullError
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from fastapi import FastAPI, HTTPException, Request

from model import hf_model_fn
from batch import BatchManagerForGeneration

app = FastAPI()

MODEL = None


class GenerationTaskReq(BaseModel):
    max_tokens: int = Field(gt=0, le=256, example=64)
    prompt: str = Field(
        min_length=1,
        example="Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:",
    )
    top_k: Optional[int] = Field(default=None, gt=0, example=50)
    top_p: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.5)
    temperature: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.7)


@app.get("/model")
def get_model():
    return MODEL


@app.post("/generation")
async def generate(data: GenerationTaskReq, request: Request):
    logger.info(
        f'{request.client.host}:{request.client.port} - "{request.method} {request.url.path}" - {data}'
    )

    inputs = tokenizer(data.prompt, truncation=True, max_length=512)
    inputs["max_tokens"] = data.max_tokens
    inputs["top_k"] = data.top_k
    inputs["top_p"] = data.top_p
    inputs["temperature"] = data.temperature
    try:
        uid = id(data)
        engine.submit(uid, inputs)
        output = await engine.wait(uid)
        output = tokenizer.decode(output, skip_special_tokens=True)
    except QueueFullError as e:
        raise HTTPException(status_code=406, detail=e.args[0])

    return {"text": output}


@app.on_event("shutdown")
async def shutdown(*_):
    engine.shutdown()
    server.should_exit = True
    server.force_exit = True
    await server.shutdown()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_cfg", type=Path, required=True)
    parser.add_argument("--tokenizer_file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint files")

    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--master_host", default="localhost")
    parser.add_argument("--master_port", type=int, default=19990)
    parser.add_argument("--rpc_port", type=int, default=19980)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--pipe_size", type=int, default=1)
    parser.add_argument("--queue_size", type=int, default=0)
    parser.add_argument("--http_host", default="0.0.0.0")
    parser.add_argument("--http_port", type=int, default=7070)
    parser.add_argument("--cache_size", type=int, default=0)
    parser.add_argument("--cache_list_size", type=int, default=1)

    args = parser.parse_args()

    MODEL = args.model_cfg.name
    logger = logging.getLogger(__name__)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(str(args.tokenizer_file))
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    engine = launch_engine(
        args.tp,
        1,
        args.master_host,
        args.master_port,
        args.rpc_port,
        hf_model_fn(model_cfg=args.model_cfg, tokenizer_file=args.tokenizer_file),
        batch_manager=BatchManagerForGeneration(
            max_batch_size=args.max_batch_size, pad_token_id=tokenizer.pad_token_id
        ),
        pipe_size=args.pipe_size,
        queue_size=args.queue_size,
    )
    config = uvicorn.Config(app, host=args.http_host, port=args.http_port)
    server = uvicorn.Server(config=config)
    server.run()
