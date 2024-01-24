"""Benchmark the latency of processing a single batch of requests."""
import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams


def profile_handler(
    profile_ctx, device_type="cpu", profile_prefix="gptj", profile_dir=f"./profile"
):
    def save_profile(profile_table, profile_path):
        print(profile_table)
        with open(profile_path, "w") as profile_file:
            profile_file.write(profile_table)

    os.makedirs(profile_dir, exist_ok=True)
    print(f"Exporting {profile_prefix} to {profile_dir}")
    save_profile(
        profile_ctx.key_averages().table(sort_by=f"self_{device_type}_time_total"),
        os.path.join(profile_dir, f"{profile_prefix}_profile.prof"),
    )
    # save_profile(
    #     profile_ctx.table(sort_by="id", row_limit=-1),
    #     os.path.join(profile_dir, f"{profile_prefix}_profile_id.prof"),
    # )

    # save_profile(
    #     profile_ctx.key_averages(group_by_input_shape=True).table(),
    #     os.path.join(profile_dir, f"{profile_prefix}_profile_group.prof"),
    # )
    profile_ctx.export_chrome_trace(
        os.path.join(profile_dir, f"{profile_prefix}_trace.json")
    )


def main(args: argparse.Namespace):
    print(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        block_size=32
    )

    # init Sampler
    if args.n == 1:
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
            use_beam_search=args.use_beam_search,
            best_of=args.n,
            top_k=-1,
            temperature=0,
            max_tokens=128,
            early_stopping=True
        )  # BeamSearch
    print(f"sampling_params:{sampling_params}", flush=True)
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    def run_to_completion(args):
        if args.profile:
            if args.device_type == "xpu":
                with torch.autograd.profiler_legacy.profile(
                    enabled=args.profile, use_xpu=(args.device_type == "xpu"), record_shapes=True
                ) as prof:
                    llm.generate(
                        prompt_token_ids=dummy_prompt_token_ids,
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )
                    torch.xpu.synchronize()
                profile_handler(prof, args.device_type, f"gptj_latency")
            else:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True
                ) as prof:
                    llm.generate(
                        prompt_token_ids=dummy_prompt_token_ids,
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )
                    torch.cuda.synchronize()
                profile_handler(prof, args.device_type, f"gptj_latency")
        else:
            start_time = time.perf_counter()
            llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    run_to_completion(args)

    if args.profile:
        run_to_completion(args)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(args))
    print(f'Avg latency: {np.mean(latencies)} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=(
            'path to save the pytorch profiler output. Can be visualized '
            'with ui.perfetto.dev or Tensorboard.'
        ))
    parser.add_argument(
        "--optimize_transformers",
        action="store_true",
        help="Enable IPEX optimize_transformers for xpu.",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        choices=["cpu", "xpu", "cuda"],
        default="xpu",
        help="Device type for inference, choose from cpu, xpu or cuda",
    )
    args = parser.parse_args()
    main(args)
