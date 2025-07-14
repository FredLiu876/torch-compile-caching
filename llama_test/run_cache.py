import torch, os, time, json, sys, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run Llama-3.1-8B with cache and prompt")
    parser.add_argument("--cache_path", required=True, help="Path to the cache file (e.g., llama8b1.megacache)")
    args = parser.parse_args()

    # Hydrate the cache before defining any compiled functions
    with open(args.cache_path, "rb") as f:
        torch.compiler.load_cache_artifacts(f.read())

    print("Mega-Cache pre-loaded — expecting instant graph start-up")

    model_name = "meta-llama/Llama-3.1-8B"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16).to("cuda")

    @torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=True)
    def llm_forward(input_ids, attention_mask=None):
        return model(input_ids=input_ids,
                     attention_mask=attention_mask).logits

    with open("prompts.txt", "r") as f:
        prompts = f.read().splitlines()

    df = pd.read_csv("test_results.csv")
    for prompt in prompts:
        print(f"Running prompt: {prompt}")
        inputs = tok(prompt, return_tensors="pt").to("cuda")

        t0 = time.time()
        out = llm_forward(**inputs)
        cache_first_run_time = time.time() - t0
        print("First run latency:", cache_first_run_time, "s")
        # Save first run latency to csv
        df.loc[df["prompt"] == prompt, "cache_first_run_time"] = cache_first_run_time

        # Run 5 more times to measure compilation performance
        for i in range(5):
            t0 = time.time()
            out = llm_forward(**inputs)
            runtime = time.time() - t0
            if i == 4:
                # Save the last run latency to csv
                print(f"Run {i+1} latency:", runtime, "s")
                df.loc[df["prompt"] == prompt, "cache_final_run_time"] = runtime
    
    df.to_csv("test_results.csv", index=False)

if __name__ == "__main__":
    main()