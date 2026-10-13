from typing import Any


def create_llm_engine(model_string: str, use_cache: bool = False, is_multimodal: bool = False, **kwargs) -> Any:
    from .openai import ChatOpenAI

    temperature_explicit = "temperature" in kwargs
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
    if "base_url" in kwargs and kwargs["base_url"] is not None:
        config["base_url"] = kwargs["base_url"]
    if "api_key" in kwargs and kwargs["api_key"] is not None:
        config["api_key"] = kwargs["api_key"]
    if "openai_compatible" in kwargs and kwargs["openai_compatible"] is not None:
        config["openai_compatible"] = kwargs["openai_compatible"]
    # Preserve explicit temperature (needed for RL exploration); only default n=1 → greedy.
    if config["n"] == 1 and not temperature_explicit:
        config["temperature"] = 0

    return ChatOpenAI(**config)
