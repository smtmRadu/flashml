QWEN3_0_6B_CONFIG_VLLM_CONFIG = {
    "model": "unsloth/Qwen3-0.6B-bnb-4bit",
    "max_model_len": 10240,
    "max_completion_tokens": 4096,
    "gpu-memory-utilization": 0.8,
    "tensor-parallel-size": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}

QWEN3_4B_THINKING_2507_VLLM_CONFIG = {
    "model": "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
    "max_model_len": 10240,
    "max_completion_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "tensor-parallel-size": 1,
    "gpu-memory-utilization": 0.85,
}

QWEN3_VL_2B_THINKING_VLLM_CONFIG = {
    "model": "unsloth/Qwen3-VL-2B-Thinking-bnb-4bit",
    "max_model_len": 8192,
    "max_completion_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "tensor-parallel-size": 1,
    "limit-mm-per-prompt.video": 0,
    "async-scheduling": "",
    "gpu-memory-utilization": 0.85,
}

QWEN3_VL_4B_THINKING_AWQ_VLLM_CONFIG = {
    "model": "cpatonn/Qwen3-VL-4B-Thinking-AWQ-4bit",
    "max_model_len": 8192,
    "max_completion_tokens": 7000,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "tensor-parallel-size": 1,
    "limit-mm-per-prompt.video": 0,
    "async-scheduling": "",
    "gpu-memory-utilization": 0.85,
}

QWEN3_VL_8B_THINKING_VLLM_CONFIG = {
    "model": "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "limit-mm-per-prompt.video": 0,
    "async-scheduling": "",
    "gpu-memory-utilization": 0.95,
    "tensor-parallel-size": 1,
}


QWEN3_VL_30B_A3B_THINKING_VLLM_CONFIG = {
    "model": "QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "limit-mm-per-prompt.video": 0,
    "async-scheduling": "",
    "gpu-memory-utilization": 0.95,
    "tensor-parallel-size": 1,
}


QWEN3_5_122B_A10B_TEXT_THINKING_VLLM_CONFIG = {
    "model": "Qwen/Qwen3.5-122B-A10B",
    "max_model_len": 81_920,
    "max_completion_tokens": 61_440,
    
    "gpu-memory-utilization": 0.95,
    "tensor-parallel-size": 1,
    "temperature": 0.9, # recommended is 1 but from my tests 0.9 is better (1 fails a little in the final answer)
    
    "top_p": 0.95,
    "top_k": 20,
    "reasoning-parser": "qwen3",
    "language-model-only": "",
    "enable-prefix-caching": ""
}

QWEN3_5_35B_A3B_TEXT_THINKING_VLLM_CONFIG = {
    "model": "Qwen/Qwen3.5-35B-A3B",
    "max_model_len": 81_920,
    "max_completion_tokens": 61_440,
    
    "gpu-memory-utilization": 0.95,
    "tensor-parallel-size": 1,
    "temperature": 0.9, # recommended is 1 but from my tests 0.9 is better (1 fails a little in the final answer)
    
    "top_p": 0.95,
    "top_k": 20,
    "reasoning-parser": "qwen3",
    "language-model-only": "",
    "enable-prefix-caching": ""
}


GEMMA3_4B_IT_VLLM_CONFIG = {
    "model": "google/gemma3-4b-it",
    "max_model_len": 32768,
    "max_completion_tokens": 16384,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 100,
    "tensor-parallel-size": 1,
    "async-scheduling": "",
    "gpu-memory-utilization": 0.85,
    "limit_mm_per_prompt.image": 0,
    "limit_mm_per_prompt.video": 0,
}


# text only btw (vision one requires preprocessor_config.json)
GEMMA3_27B_IT_VLLM_CONFIG = {
    "model": "google/gemma3-27b-it",
    "max_model_len": 32768,
    "max_completion_tokens": 16384,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 100,
    "tensor-parallel-size": 1,
    "async-scheduling": "",
    "gpu-memory-utilization": 0.9,
    "limit_mm_per_prompt.image": 0,
    "limit_mm_per_prompt.video": 0,
}



GPT_OSS_120B_HIGH_VLLM_CONFIG = {
    "model": "openai/gpt-oss-120b",
    "max_model_len": 81_920, # the model reasons even 12.5k samples
    "max_completion_tokens": 61_440,
    "gpu-memory-utilization": 0.9,
    "tensor-parallel-size": 1,
    "temperature": 1.0, # recommended is 1 but from my tests 0.9 is better (1 fails a little in the final answer)
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "high",
    "ignore-patterns": ["original/**", "metal/**"]
    # don't put async scheduling, it fucks up the memory..
}

GPT_OSS_120B_LOW_VLLM_CONFIG = {
    "model": "openai/gpt-oss-120b",
    "max_model_len": 81_920, # the model reasons even 12.5k samples
    "max_completion_tokens": 61_440,
    "gpu-memory-utilization": 0.9,
    "tensor-parallel-size": 1,
    "temperature": 1.0, # recommended is 1 but from my tests 0.9 is better (1 fails a little in the final answer)
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "low",
    "ignore-patterns": ["original/**", "metal/**"]
    # don't put async scheduling, it fucks up the memory..
}


GPT_OSS_20B_LOW_VLLM_CONFIG = {
    "model": "openai/gpt-oss-20b",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "reasoning_effort": "low",
    "gpu-memory-utilization": 0.95,
    "tensor-parallel-size": 1,
    "temperature": 1.0,
    "top_p": 1,
    "top_k": -1,
    "ignore-patterns": ["original/**", "metal/**"]
    # don't put async scheduling, it fucks up the memory..
}

MINISTRAL_3_3B_INSTRUCT_2512_FINETUNED_VLLM_CONFIG = {
    "max_model_len": 4096,
    "tensor-parallel-size": 1,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 128,
    #"quantization" : "bitsandbytes", #in practice not ok to quantize at runtime.. but because i didn't manage to quantize it keep it like this.
    "gpu-memory-utilization": 0.85,
    "tokenizer-mode": "mistral",
    "config_format": "auto",
    "load_format": "safetensors",
}

MINISTRAL_3_3B_INSTRUCT_2512_VLLM_CONFIG = {
    "model": "mistralai/Ministral-3-3B-Instruct-2512",
    "max_model_len": 8192,
    "max_completion_tokens": 4096,
    "gpu-memory-utilization": 0.85,
    "tensor-parallel-size": 1,
    "temperature": 0.1,
    "tokenizer-mode": "mistral",
    "config_format": "mistral",
    "load_format": "mistral",
    "ignore_patterns": ["original/**", "metal/**"],
}


