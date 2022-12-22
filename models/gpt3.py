import time
import os
import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError

api_keys = os.getenv("OPENAI_API_KEYS").split(",")
cur_key_idx = 0

delay_time = 5
decay_rate = 0.8

def gpt3_completion(prompts, model="code-davinci-002", max_tokens=400):
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
            n=1,
            stop=["\n"]
        )
        delay_time *= decay_rate
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError) as exc:
        # print(exc)
        delay_time *= 2
        return gpt3_completion(prompts, model, max_tokens)

    # Extract text from response
    return [choice["text"] for choice in response["choices"]]
