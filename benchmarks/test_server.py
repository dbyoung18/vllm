import asyncio
import time

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
# initialize the engine and the example input


async def infer():
    engine_args = AsyncEngineArgs(
        model="EleutherAI/gpt-j-6B",
        block_size=32,
        enforce_eager=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    example_input = {
        # "prompt": "What is LLM?",
        "prompt": "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
        "stream": True, # assume the non-streaming case
        "temperature": 0.0,
        "request_id": 0,
    }
    time.sleep(60)

    import torch
    import intel_extension_for_pytorch as ipex
    engine.engine.driver_worker.model_runner.model = ipex.optimize_transformers(
        engine.engine.driver_worker.model_runner.model.to(torch.float16), dtype="float16", inplace=True, device="xpu"
    )

    num_beams = 1
    if num_beams == 1:
        print("==> use GreedySearch")
        sampling_params = SamplingParams(
            best_of=1,
            top_p=1,
            top_k=-1,
            temperature=0
        )  # GreedySearch
    else:
        print("==> use BeamSearch")
        sampling_params = SamplingParams(
            use_beam_search=True,
            best_of=num_beams,
            top_k=-1,
            temperature=0,
            max_tokens=128,
            early_stopping=True
        )  # BeamSearch
    print(f"sampling_params:{sampling_params}", flush=True)

    # start the generation
    results_generator = engine.generate(
        prompt = example_input["prompt"],
        sampling_params = sampling_params,
        request_id = 1,
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        # if await request.is_disconnected():
        #     # Abort the request if the client disconnects.
        #     await engine.abort(request_id)
        #     # Return or raise an error
        final_output = request_output
        # print(final_output)

if __name__ == "__main__":
    asyncio.run(infer())