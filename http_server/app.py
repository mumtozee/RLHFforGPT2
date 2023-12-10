from functools import partial
import typing as tp

import fastapi
from fastapi.exceptions import HTTPException
from fastapi.responses import PlainTextResponse
import grpc
from loguru import logger
from proto_lib import inference_server_pb2, inference_server_pb2_grpc
from pydantic import BaseModel


class InputRequest(BaseModel):
    prompt: str
    generation_params: tp.Dict[str, tp.Any]


class OutputResponse(BaseModel):
    generated_text: str
    usage_information: tp.Dict[str, int]

async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

def generate(request: InputRequest, grpc_port: int) -> OutputResponse:
    logger.info(
        f"Prompt: {request.prompt} with generation_params: {request.generation_params}"
    )
    channel = grpc.insecure_channel(f"localhost:{grpc_port}")
    stub = inference_server_pb2_grpc.GPT2ServiceStub(channel)
    try:
        grpc_response = stub.Generate(
            inference_server_pb2.PromptRequest(
                prompt=request.prompt, generation_params=request.generation_params
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EXCEPTION OCCURRED: {str(e)}")

    return OutputResponse(
        generated_text=grpc_response.generated_text,
        usage_information=grpc_response.usage_information,
    )


def get_app(grpc_port: int = 50051) -> fastapi.FastAPI:
    app = fastapi.FastAPI()

    app.add_api_route(
        "/generate",
        partial(generate, grpc_port=grpc_port),
        methods=["POST"],
        response_model=OutputResponse,
    )
    app.add_exception_handler(HTTPException, http_exception_handler)

    return app
