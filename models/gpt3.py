from typing import List
import time
import os
import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError

api_keys = os.getenv("OPENAI_API_KEYS").split(",")
cur_key_idx = 0

delay_time = 5
decay_rate = 0.8

BATCH_SIZE = 10

def gpt3_completion_with_batching(prompts: List[str], model="code-davinci-002", max_tokens=400, logprobs=None, echo=False):
    results = []
    for start_idx in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start_idx : start_idx + BATCH_SIZE]
        results += gpt3_completion(batch, model, max_tokens, logprobs, echo)
    return results

def gpt3_completion(prompts: List[str], model="code-davinci-002", max_tokens=400, logprobs=None, echo=False):
    global delay_time, cur_key_idx
    time.sleep(delay_time)
    # print(f"{delay_time:.3f}")

    # Alternate keys
    cur_key_idx = (cur_key_idx + 1) % len(api_keys)
    openai.api_key = api_keys[cur_key_idx]

    # Send request
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompts,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
            logprobs=logprobs,
            echo=echo
        )
        delay_time *= decay_rate
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError) as exc:
        # print(exc)
        delay_time *= 2
        return gpt3_completion(prompts, model, max_tokens)

    return response["choices"]
