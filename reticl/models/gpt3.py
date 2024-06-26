from typing import List
import time
import os
import math
import random
import concurrent.futures
import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError

api_keys = os.getenv("OPENAI_API_KEYS").split(",")
cur_key_idx = 0

delay_time = 0.5
decay_rate = 0.8

def gpt3_completion_with_batching(prompts: List[str], max_batch_size=5, model="code-davinci-002", max_tokens=400, logprobs=None, echo=False, verbose=False):
    # Break up requests evenly among keys, using largest batch size possible
    num_batches = math.ceil(len(prompts) / max_batch_size)
    batch_size = math.ceil(len(prompts) / num_batches)

    results = []
    for start_idx in range(0, len(prompts), batch_size):
        batch = prompts[start_idx : start_idx + batch_size]
        results += gpt3_completion(batch, model, max_tokens, logprobs, echo, verbose=verbose)
    return results

def gpt3_completion_parallel(prompts: List[str], max_batch_size=5, model="code-davinci-002", max_tokens=400, logprobs=None, echo=False, verbose=False):
    global cur_key_idx

    # Break up requests evenly among keys, using largest batch size possible
    num_batches = math.ceil(len(prompts) / max_batch_size)
    batch_size = math.ceil(len(prompts) / num_batches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_batches) as executor:
        # Submit batches to threads
        futures = []
        for start_idx in range(0, len(prompts), batch_size):
            batch = prompts[start_idx : start_idx + batch_size]
            cur_key_idx = (cur_key_idx + 1) % len(api_keys)
            futures.append(executor.submit(gpt3_completion, batch, model, max_tokens, logprobs, echo, api_keys[cur_key_idx], verbose))

        # Wait for all to complete
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        # Accumulate results
        results = [result for future in futures for result in future.result()]
        return results

def gpt3_completion(prompts: List[str], model="code-davinci-002", max_tokens=400, logprobs=None, echo=False, key_to_use=None, verbose=False):
    global delay_time, cur_key_idx

    # Wait for rate limit, add random jitter to avoid thread collisions
    if model != "gpt-3.5-turbo":
        time.sleep(delay_time + random.random() * 2e-2)
    # print(f"{delay_time:.3f}", key_to_use)

    # Assign/cycle API key
    if key_to_use:
        openai.api_key = key_to_use
    else:
        cur_key_idx = (cur_key_idx + 1) % len(api_keys)
        openai.api_key = api_keys[cur_key_idx]

    # Send request
    try:
        if model in ("gpt-3.5-turbo", "gpt-4"):
            results = []
            for prompt in prompts:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    request_timeout=45
                )
                results.append(response["choices"][0])
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prompts,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n\n"],
                logprobs=logprobs,
                echo=echo
            )
            results = response["choices"]
        delay_time = max(delay_time * decay_rate, 0.1)
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError) as exc:
        if verbose:
            print(openai.api_key, exc)
        delay_time = min(delay_time * 2, 30)
        return gpt3_completion(prompts, model, max_tokens, logprobs, echo, key_to_use)

    return results
