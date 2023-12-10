import asyncio
import threading
from concurrent import futures
import typing as tp
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
from utils import CustomStopCriteria

import grpc
from absl import app, flags  # CLI INTERFACE
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from loguru import logger
import uvicorn

from http_server import app as fastapi_app
from proto_lib import inference_server_pb2, inference_server_pb2_grpc

# CLI PARAMETERS, try `python main.py --help`
FLAGS = flags.FLAGS
_GRPC_PORT = flags.DEFINE_integer("grpc_port", 50051, "Port to serve grpc on")
_FASTAPI_PORT = flags.DEFINE_integer("fastapi_port", None, "Port to serve fastapi on")


#### GRPC IMPLEMENTATION ####
class GPT2ServerImplementation(inference_server_pb2_grpc.GPT2ServiceServicer):
    def __init__(self, type: str = "hh:ppo"):
        super().__init__()
        self.dataset_type, self.alignment_type = type.split(":")
        suffix = "" if self.dataset_type == "hh" else "_caps"
        model_path = f"./{self.alignment_type}_checkpoints{suffix}"
        logger.info(f"Loading model from: {model_path}")
        self.pipe = pipeline(task="text-generation", model=model_path, device="cuda:0")

    def Generate(self, request: inference_server_pb2.PromptRequest, context):
        logger.info(
            f"Prompt: [{request.prompt}] with generation_params: {request.generation_params}"
        )
        gen_params = self.validate_params(request.generation_params)
        toker = self.pipe.tokenizer
        stop_sequences = gen_params.pop("stop_sequences", None)
        if stop_sequences:
            stopping_criteria = StoppingCriteriaList(
                [
                    CustomStopCriteria(custom_tokens=toker(x)["input_ids"])
                    for x in stop_sequences
                ]
            )
        else:
            stopping_criteria = StoppingCriteriaList(
                [CustomStopCriteria(uppercase=self.dataset_type == "alpaca")]
            )
        gen_params["stopping_criteria"] = stopping_criteria
        gen_params["return_full_text"] = False
        gen_res = self.pipe(request.prompt, **gen_params)[0]["generated_text"]
        num_tokens_in_prompt = len(toker(request.prompt)["input_ids"])
        num_generated_tokens = len(toker(gen_res)["input_ids"])
        result = inference_server_pb2.PromptResponse(
            generated_text=gen_res,
            usage_information={
                "prompt_tokens": num_tokens_in_prompt,
                "generated_tokens": num_generated_tokens,
                "total": num_tokens_in_prompt + num_generated_tokens,
            },
        )
        return result

    def validate_params(self, gen_params: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        new_params = {}
        if "top_k" in gen_params:
            new_params["top_k"] = int(gen_params["top_k"])
        if "top_p" in gen_params:
            new_params["top_p"] = float(gen_params["top_p"])
        if "temperature" in gen_params:
            new_params["temperature"] = float(gen_params["temperature"])
        if "repetetion_penalty" in gen_params:
            new_params["repetition_penalty"] = float(gen_params["repetition_penalty"])
        if "max_new_tokens" in gen_params:
            new_params["max_new_tokens"] = int(gen_params["max_new_tokens"])
        if "stop_sequences" in gen_params:
            new_params["stop_sequences"] = gen_params["stop_sequences"].split(";")
        return new_params


def serve_grpc(port: int, gen_type: str):
    async def serve_inner(port: int, gen_type: str):
        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        inference_server_pb2_grpc.add_GPT2ServiceServicer_to_server(
            GPT2ServerImplementation(gen_type), server
        )
        server.add_insecure_port(f"[::]:{port}")

        #### REFLECTIONS ####
        SERVICE_NAMES = (
            inference_server_pb2.DESCRIPTOR.services_by_name["GPT2Service"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        #### DONE REFLECTIONS ####

        #### HEALTH CHECK ####
        health_servicer = health.HealthServicer()

        # Initialize health status
        health_servicer.set(
            "", health_pb2.HealthCheckResponse.SERVING
        )  # '' denotes the overall status of the server
        health_servicer.set(
            "gpt2_service.GPT2Service", health_pb2.HealthCheckResponse.SERVING
        )

        # Register health service
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        #### DONE HEALTH CHECK ####

        logger.info(f"Starting grpc server on port {port}")
        await server.start()

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(serve_inner(port, gen_type))


#### GRPC IMPLEMENTATION ####


def main(argv: tp.Sequence) -> None:
    if len(argv) > 1:
        gen_type = argv[1]
    else:
        gen_type = "hh:dpo"
    # Do not do this in real word, this is just for demo purposes
    grpc_thread = threading.Thread(
        target=serve_grpc,
        args=(
            _GRPC_PORT.value,
            gen_type,
        ),
    )
    grpc_thread.start()

    app = fastapi_app.get_app()
    uvicorn.run(app, host="0.0.0.0", port=_FASTAPI_PORT.value)


if __name__ == "__main__":
    flags.mark_flag_as_required("fastapi_port")
    app.run(main)
