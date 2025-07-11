test_prompts = [
    "Hello!",
    "What is the capital of France?",
    "Explain the concept of recursion.",
    "List three benefits of regular exercise.",
    "Write a Python function to reverse a string.",
    "Describe the process of photosynthesis in plants.",
    "Summarize the plot of your favorite movie in two sentences.",
    "How does the internet work? Provide a brief overview, including the roles of servers, clients, and protocols such as HTTP and TCP/IP.",
    "Compare and contrast object-oriented and functional programming paradigms, discussing their principles, advantages, disadvantages, and typical use cases in modern software development.",
    "Imagine you are planning a trip to Japan. What are the top five places you would like to visit and why? Additionally, outline a detailed itinerary for a two-week stay, including cultural experiences, local cuisine you want to try, and any language or travel challenges you anticipate."
]

import torch, os

# Create a csv with a line for each prompt
# First column is time for first run, second column is time for the final run
# Using df
import pandas as pd
df = pd.DataFrame(columns=["prompt", "no_compile_first_run_time", "no_compile_final_run_time", "compile_first_run_time", "compile_final_run_time", "cache_first_run_time", "cache_final_run_time"])
# Create a row for each prompt and initialize the times to 0
for i, prompt in enumerate(test_prompts):
    df.loc[i] = [prompt, 0, 0, 0, 0, 0, 0]
# Save the dataframe to a csv file
df.to_csv("test_results.csv", index=False)

for i, prompt in enumerate(test_prompts):
    print(f"Test {i+1}: {prompt}")

    # Run the run_load_from_compile.py script with the current prompt and print the output of the script
    os.system(f"python3 run_load_from_compile.py --prompt \"{prompt}\" --cache_path \"compiled_model_cache{i}\"")

for i, prompt in enumerate(test_prompts):
    print(f"Test {i+1}: {prompt}")

    # Run the run_load_from_compile.py script with the current prompt and print the output of the script
    os.system(f"python3 run_load_from_cache.py --prompt \"{prompt}\" --cache_path \"compiled_model_cache{i}\"")

for i, prompt in enumerate(test_prompts):
    print(f"Test {i+1}: {prompt}")

    # Run the run.py script with the current prompt and print the output of the script
    os.system(f"python3 run.py --prompt \"{prompt}\"")
