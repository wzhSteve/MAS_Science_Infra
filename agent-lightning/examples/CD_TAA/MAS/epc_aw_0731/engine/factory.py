from typing import Any
import os


def create_llm_engine(model_string: str, use_cache: bool = False, is_multimodal: bool = False, **kwargs) -> Any:
    
    # print(f"creating llm engine {model_string} with: is_multimodal: {is_multimodal}, kwargs: {kwargs}")

    from .openai import ChatOpenAI
    config = {
        "model_string": model_string,
        "use_cache": use_cache,
        "is_multimodal": is_multimodal,
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9),
        "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
        "presence_penalty": kwargs.get("presence_penalty", 0.5),
        "n": kwargs.get("n", 1),
    }
    if config["n"] == 1:
        config["temperature"] = 0

    return ChatOpenAI(**config)