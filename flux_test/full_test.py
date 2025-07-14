import torch, os

# Create a csv with a line for each prompt
# First column is time for first run, second column is time for the final run
# Using df
import pandas as pd
df = pd.DataFrame(columns=["prompt", "no_compile_first_run_time", "no_compile_final_run_time", "compile_first_run_time", "compile_final_run_time", "cache_first_run_time", "cache_final_run_time"])
# Create a row for each prompt and initialize the times to 0
with open("prompts.txt", "r") as f:
    test_prompts = f.read().splitlines()
for i, prompt in enumerate(test_prompts):
    df.loc[i] = [prompt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Save the dataframe to a csv file
df.to_csv("test_results.csv", index=False)

# for i, prompt in enumerate(test_prompts):
#     print(f"Test {i+1}: {prompt}")

#     # Run the run_load_from_compile.py script with the current prompt and print the output of the script
#     os.system(f"python3 run_load_from_compile.py --prompt \"{prompt}\" --cache_path \"compiled_model_cache{i}\"")

# for i, prompt in enumerate(test_prompts):
#     print(f"Test {i+1}: {prompt}")

#     # Run the run_load_from_compile.py script with the current prompt and print the output of the script
#     os.system(f"python3 run_load_from_cache.py --prompt \"{prompt}\" --cache_path \"compiled_model_cache{i}\"")

# for i, prompt in enumerate(test_prompts):
#     print(f"Test {i+1}: {prompt}")

#     # Run the run.py script with the current prompt and print the output of the script
#     os.system(f"python3 run.py --prompt \"{prompt}\"")
