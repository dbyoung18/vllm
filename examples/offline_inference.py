from vllm import LLM, SamplingParams
import torch
import intel_extension_for_pytorch as ipex
import time
import os
# Sample prompts.
prompts = [
    "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.9)

# Create an LLM.
llm = LLM(model="EleutherAI/gpt-j-6B")
print(f"==> num workers:{len(llm.llm_engine.workers)}")
for idx, worker in enumerate(llm.llm_engine.workers):
    llm.llm_engine.workers[idx].model = ipex.optimize_transformers(worker.model, dtype=torch.float16, inplace=True, device="xpu")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
profile_enabled = os.environ.get("PROFILE", "OFF").upper() in ["1", "ON", "YES", "TRUE"]

# for warm up
for i in range(5):
    outputs = llm.generate(prompts, sampling_params)

for i in range(10):
    tic = time.time()
    with torch.autograd.profiler_legacy.profile(enabled=profile_enabled, use_xpu=True, record_shapes=True) as prof:
        outputs = llm.generate(prompts, sampling_params)
        torch.xpu.synchronize()
    if profile_enabled:
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), "./profile.pt")
        torch.save(prof.table(sort_by="id", row_limit=-1),'./profile_id.pt')
        torch.save(prof.key_averages(group_by_input_shape=True).table(), "./profile_detail.pt")
        prof.export_chrome_trace("./trace.json")
    e2e = time.time() - tic
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}", flush=True)
    print("Iteration: %d, Time: %.6f sec" % (i, e2e), flush=True)


