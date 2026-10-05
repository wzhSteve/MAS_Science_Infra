"""Patch EPC-AW engine factory to support local VLLM.

This module patches the factory to automatically detect and use VLLM
when OPENAI_API_BASE points to localhost.
"""

import os
from typing import Any


def patch_factory_for_vllm() -> None:
    """Patch create_llm_engine to use VLLM when running locally."""
    try:
        from MAS.epc_aw.engine import factory as epc_factory
    except ImportError:
        print("[WARN] Cannot import MAS.epc_aw.engine.factory, skipping patch")
        return
    
    # Check if we should use VLLM
    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VLLM_BASE_URL")
        or ""
    ).strip()
    
    use_vllm = (
        "localhost" in base_url.lower() 
        or "127.0.0.1" in base_url.lower()
        or base_url.startswith("http://")
    )
    
    if not use_vllm:
        print(f"[INFO] Not using VLLM, base_url={base_url}")
        return
    
    print(f"[INFO] Detected local LLM at {base_url}, patching factory to use VLLM...")
    
    original_create = epc_factory.create_llm_engine
    if getattr(original_create, "_science_vllm_base", None) == base_url:
        return

    def patched_create(
        model_string: str,
        use_cache: bool = False,
        is_multimodal: bool = False,
        **kwargs
    ) -> Any:
        """Create LLM engine with VLLM support for local models."""
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
        
        # Try to use VLLM for local models
        if use_vllm:
            try:
                from MAS.epc_aw.engine.vllm import ChatVLLM
                
                # Map environment variables for VLLM
                vllm_base = base_url or "http://localhost:8000/v1"
                vllm_key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy-token"
                
                config["base_url"] = vllm_base
                config["api_key"] = vllm_key
                
                print(f"[INFO] Creating VLLM engine: model={model_string}, base_url={vllm_base}")
                return ChatVLLM(**config)
            
            except ImportError as e:
                print(f"[WARN] VLLM not available ({e}), falling back to OpenAI")
            except Exception as e:
                print(f"[WARN] Failed to create VLLM engine ({e}), falling back to OpenAI")
        
        # Fallback to OpenAI
        from MAS.epc_aw.engine.openai import ChatOpenAI
        print(f"[INFO] Creating OpenAI-compatible engine: model={model_string}, base_url={base_url or 'default'}")
        return ChatOpenAI(**config)

    patched_create._science_vllm_base = base_url  # type: ignore[attr-defined]
    epc_factory.create_llm_engine = patched_create
    print("[INFO] Factory patched successfully")


if __name__ == "__main__":
    patch_factory_for_vllm()
