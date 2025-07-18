# flux_megacache_bench.py
import argparse, os, time, json, pickle, torch
import pandas as pd
from diffusers import FluxPipeline

MODEL_ID   = "black-forest-labs/FLUX.1-schnell"
DTYPE      = torch.float16
DEVICE     = "cuda"
CACHE_FILE = "flux_schnell.megacache"
PROMPT     = "A neon cyber-punk cityscape at dusk"
SEED       = 1234          # needed for determinism

def sync():  # helper for accurate timings
    if torch.cuda.is_available(): torch.cuda.synchronize()

def timed(tag, fn, *a, **kw):
    sync(); t0 = time.perf_counter()
    out = fn(*a, **kw)
    sync(); dt = time.perf_counter() - t0
    print(f"{tag:40s}{dt:8.4f} s")
    return out, dt

def init_pipe(compile=False):
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    if compile:
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune-no-cudagraphs", dynamic=False
        )

        pipe.vae.decode = torch.compile(
            pipe.vae.decode, mode="max-autotune-no-cudagraphs", dynamic=False
        )
    return pipe

def sample_latents(pipe, prompt):
    generator = torch.Generator(DEVICE).manual_seed(SEED)
    out = pipe(
        prompt=prompt,
        num_inference_steps=4,
        width=512,
        height=512,
        output_type="latent",
        generator=generator,
    )
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["run_compile", "run_cache", "run"])
    args = p.parse_args()

    with open("prompts.txt", "r") as f:
        prompts = f.read().splitlines()

    df = pd.read_csv("test_results.csv")
    for prompt in prompts:
        # ---------------------------------------------------------------- build ---
        if args.phase == "run_compile":
            pipe = init_pipe(True)

            _, warm_dt = timed("compile + warm-up (includes codegen)", sample_latents, pipe, prompt)
            lat, post_dt = timed("post-compile inference", sample_latents, pipe, prompt)

            df.loc[df["prompt"] == prompt, "compile_first_run_time"] = warm_dt
            df.loc[df["prompt"] == prompt, "compile_final_run_time"] = post_dt

            blob, meta = torch.compiler.save_cache_artifacts()
            open(CACHE_FILE, "wb").write(blob)
            print(f"\nSaved Mega-Cache → {CACHE_FILE}")

        # ---------------------------------------------------------------- run -----
        elif args.phase == "run_cache":
            torch.compiler.load_cache_artifacts(open(CACHE_FILE, "rb").read())
            print("Mega-Cache hydrated — expect instant start-up\n")

            pipe = init_pipe(True)

            _, warm_dt = timed("loaded_cache warm-up (includes codegen)", sample_latents, pipe, prompt)
            lat, post_dt = timed("post-compile inference", sample_latents, pipe, prompt)

            df.loc[df["prompt"] == prompt, "cache_first_run_time"] = warm_dt
            df.loc[df["prompt"] == prompt, "cache_final_run_time"] = post_dt

        # ---------------------------------------------------------------- run -----
        elif args.phase == "run":
            pipe = init_pipe()

            _, eager_dt = timed("eager 1-shot (winds up kernels)", sample_latents, pipe, prompt)
            _, eager2_dt = timed("eager 1-shot (winds up kernels) 2", sample_latents, pipe, prompt)
            df.loc[df["prompt"] == prompt, "no_compile_first_run_time"] = eager_dt
            df.loc[df["prompt"] == prompt, "no_compile_final_run_time"] = eager2_dt

        else:
            raise ValueError(f"Unknown phase: {args.phase}")
    df.to_csv("test_results.csv", index=False)