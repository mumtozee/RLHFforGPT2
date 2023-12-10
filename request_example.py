import argparse
import aiohttp
import asyncio
import httpx
from pprint import pprint
import json
import requests

FASTAPI_PORT = 8898
URL = f"http://127.0.0.1:{FASTAPI_PORT}/generate?grpc_port=50051"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="all")
    args = parser.parse_args()
    return args


async def aio_req(prompt, gen_params):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            URL,
            json={
                "prompt": prompt,
                "generation_params": gen_params,
            },
        ) as response:
            print(f"Status: {response.status}")
            pprint(await response.json())


def httpx_req(prompt, gen_params):
    r = httpx.post(
        URL,
        json={
            "prompt": prompt,
            "generation_params": gen_params,
        },
    )
    pprint(json.loads(r.text))


def requests_req(prompt, gen_params):
    r = requests.post(
        URL,
        json={
            "prompt": prompt,
            "generation_params": gen_params,
        },
    )
    pprint(json.loads(r.text))


async def main() -> None:
    args = parse_args()
    lib = args.library
    prompt = "\n\nHuman: Hey, can you help me with chemistry?\n\nAssistant:"
    gen_params = {
        "top_k": "100",
        "temperature": "0.8",
        "top_p": "0.92",
        "max_new_tokens": "256",
        # "stop_sequences": "\n\nHuman:",
    }
    if lib == "asyncio":
        await aio_req(prompt, gen_params)
    elif lib == "httpx":
        httpx_req(prompt, gen_params)
    elif lib == "requests":
        requests_req(prompt, gen_params)
    elif lib == "all":
        await aio_req(prompt, gen_params)
        httpx_req(prompt, gen_params)
        requests_req(prompt, gen_params)
    else:
        raise ValueError("Invalid library for io.")


asyncio.run(main())
