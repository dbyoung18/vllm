import argparse
import json
import time
import torch
import os

from vllm import LLM, SamplingParams


# Sample prompts.
prompts = [
    "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."  # 32in
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

def parse_args():
    # Create a sampling params object.
    parser = argparse.ArgumentParser(description="Demo on vLLM for GPT-J.")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-j-6B", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default="data/prompt.json", help="Dataset path.")
    parser.add_argument("--input_len", type=int, default=32, help="Input prompt length.")
    parser.add_argument("--profile", action="store_true", help="Enable profiler.")
    parser.add_argument(
        "--optimize_transformers",
        action="store_true",
        help="Enable IPEX optimize_transformers for xpu.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["float32", "bfloat16", "float16", "int4"],
        default="float16",
        help="Data type of the model, choose from float32, bfloat16, float16 or int4",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        choices=["cpu", "xpu", "cuda"],
        default="xpu",
        help="Device type for inference, choose from cpu, xpu or cuda",
    )
    parser.add_argument(
        "--num_beams", type=int, default=4, help="Beam width for BeamSearch, 1 for GreedySearch"
    )
    parser.add_argument("--warmup", action="store_true", help="Enable warmup")
    args = parser.parse_args()
    return args

def load_dataset(dataset_path, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    with open(dataset_path, 'r') as f:
        dataset_json = json.load(f)

    dataset = {}
    for sample in dataset_json:
        input_sample = tokenizer(
            sample["input"],
            truncation=True,
            max_length=1919,
            return_tensors=None,
            padding=True,
        )
        prompt = sample["input"]
        prompt_ids = input_sample.input_ids
        attn_masks = input_sample.attention_mask
        prompt_len = len(prompt_ids)
        dataset[int(sample["input_len"])] = (prompt, prompt_ids, attn_masks, prompt_len)
    return dataset

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

if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")

    # init Sampler
    if args.num_beams == 1:
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
            best_of=args.num_beams,
            top_k=-1,
            temperature=0,
            max_tokens=128,
            early_stopping=True
        )  # BeamSearch
    print(f"sampling_params:{sampling_params}", flush=True)

    # init Engine
    llm = LLM(model=args.model, block_size=32, enforce_eager=True)
    if args.device_type == "xpu" and args.optimize_transformers:
        import intel_extension_for_pytorch as ipex
        llm.llm_engine.driver_worker.model_runner.model = ipex.optimize_transformers(
            llm.llm_engine.driver_worker.model_runner.model, dtype=args.data_type, inplace=True, device=args.device_type
        )

        for idx, worker in enumerate(llm.llm_engine.workers):
            llm.llm_engine.workers[idx].model = ipex.optimize_transformers(
                worker.model_runner.model, dtype=args.data_type, inplace=True, device=args.device_type
            )

    # init Dataset
    if args.dataset_path and args.input_len:
        dataset = load_dataset(args.dataset_path, llm.llm_engine.tokenizer)
        prompts = [dataset[int(args.input_len)][0]] * args.batch_size
        prompt_ids = [dataset[int(args.input_len)][1]] * args.batch_size
    else:
        prompts = prompts * args.batch_size
        prompt_ids = None

    if args.warmup:
        for i in range(5):
            outputs = llm.generate(prompts, sampling_params)

    # inference
    for i in range(1):
        tic = time.time()
        if args.profile:
            if args.device_type == "xpu":
                with torch.autograd.profiler_legacy.profile(
                    enabled=args.profile, use_xpu=(args.device_type == "xpu"), record_shapes=True
                ) as prof:
                    outputs = llm.generate(
                        prompts=prompts,
                        # prompt_token_ids=prompt_ids,
                        sampling_params=sampling_params
                    )
                    torch.xpu.synchronize()
                profile_handler(prof, args.device_type, f"gptj_iter{i}")
            else:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
                ) as prof:
                    outputs = llm.generate(
                        prompts=prompts,
                        # prompt_token_ids=prompt_ids,
                        sampling_params=sampling_params
                    )
                    torch.cuda.synchronize()
                profile_handler(prof, args.device_type, f"gptj_iter{i}")
        else:
            outputs = llm.generate(
                prompts=prompts,
                # prompt_token_ids=prompt_ids,
                sampling_params=sampling_params
            )
        e2e = time.time() - tic

        for output in outputs:
            prompt = output.prompt
            print(f"Prompt: {prompt}\nGenerated text: {output.outputs[0].text}\nGenerated tokens: {output.outputs[0].token_ids}", flush=True)
            print('-'*80)
        print(f"iter,{i},bs,{len(prompts)},in_len,{len(prompt_ids[0])},out_len,{len(output.outputs[0].token_ids)},time,{e2e}")
        print('='*80)
