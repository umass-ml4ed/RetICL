import time
import os
import openai
from openai.error import RateLimitError, Timeout, APIError

key_1 = os.getenv("OPENAI_API_KEY_1")
key_2 = os.getenv("OPENAI_API_KEY_2")
openai.api_key = key_1

def gpt3_completion(prompt, model="text-ada-001", max_tokens=200):
    if openai.api_key == key_1 and key_2:
        openai.api_key = key_2
    else:
        openai.api_key = key_1
    if model.startswith("code"):
        # Delay to not exceed API limit
        time.sleep((1.5 if key_2 else 2.5) * len(prompt))
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"]
        )
    except (RateLimitError, Timeout, APIError) as exc:
        print(f"{exc} - sleep then retry")
        time.sleep(30)
        return gpt3_completion(prompt, model, max_tokens)
    return [choice["text"] for choice in response["choices"]]
