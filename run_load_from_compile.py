import torch, os, time, json, sys, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run Llama-3.1-8B with cache and prompt")
    parser.add_argument("--prompt", required=True, help="Prompt to run through the model")
    parser.add_argument("--cache_path", default="compiled_model_cache",
                        help="Path to save the compiled model cache")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-3.1-8B"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16).to("cuda")

    @torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)
    def llm_forward(input_ids, attention_mask=None):
        return model(input_ids=input_ids,
                     attention_mask=attention_mask).logits

    inputs = tok(args.prompt, return_tensors="pt").to("cuda")

    t0 = time.time()
    out = llm_forward(**inputs)
    compile_first_run_time = time.time() - t0
    print("First run latency:", compile_first_run_time, "s")
    # Save first run latency to csv
    df = pd.read_csv("test_results.csv")
    df.loc[df["prompt"] == args.prompt, "compile_first_run_time"] = compile_first_run_time

    # Run 5 more times to measure compilation performance
    for i in range(5):
        t0 = time.time()
        out = llm_forward(**inputs)
        runtime = time.time() - t0
        print(f"Run {i+1} latency:", runtime, "s")
        if i == 4:
            # Save the last run latency to csv
            df.loc[df["prompt"] == args.prompt, "compile_final_run_time"] = runtime

    df.to_csv("test_results.csv", index=False)

    # Save the compiled model
    artifacts = torch.compiler.save_cache_artifacts()
    assert artifacts is not None
    artifact_bytes, cache_info = artifacts
    with open(args.cache_path, 'wb') as f:
        f.write(artifact_bytes)

if __name__ == "__main__":
    main()