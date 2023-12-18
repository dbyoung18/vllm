import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams


def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.model = "EleutherAI/gpt-j-6B"
    engine_args.tokenizer = "EleutherAI/gpt-j-6B"
    print(f"--engine_args:{engine_args}")
    engine = LLMEngine.from_engine_args(engine_args)

    # Create a sampling params object.
    # sampling_params = SamplingParams(temperature=0.9)
    if args.greedy:
        print("==> use GreedySearch")
        sampling_params = SamplingParams(
            best_of=1, top_p=1, top_k=-1, temperature=0
        )  # GreedySearch
    else:
        print("==> use BeamSearch")
        sampling_params = SamplingParams(
            use_beam_search=True, best_of=4, top_k=-1, temperature=0
        )  # BeamSearch
    # Test the following prompts.
    test_prompts = [
        ("A robot may not injure a human being", sampling_params),
        ("To be or not to be,", sampling_params),
        ("What is the meaning of life?", sampling_params),
        ("It is only with the heart that one can see rightly", sampling_params),
    ]

    # Run the engine by calling `engine.step()` manually.
    request_id = 0
    cnt = 0
    while True:
        # To test continuous batching, we add one request at each step.
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1
        cnt += 1
        request_outputs = engine.step()
        for i, request_output in enumerate(request_outputs):
            print(f"cnt,{cnt},idx,{i},req,{request_output}")
            print("-" * 30)
        print("=" * 30)

        for i, request_output in enumerate(request_outputs):
            if request_output.finished:
                print(request_output)

        if not (engine.has_unfinished_requests() or test_prompts):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument("--greedy", action="store_true", help="Enable GreedySearch")
    args = parser.parse_args()
    main(args)
