QWEN3_06B_DEFAULT_ARGS = {
    "model": "unsloth/Qwen3-0.6B-bnb-4bit",
    "max_model_len": 10240,
    "max_completion_tokens": 4096,
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}
QWEN3_4B_THINKING_2507_DEFAULT_ARGS = {
    "model": "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
    "max_model_len": 8192,
    "max_completion_tokens": 4096,
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}
QWEN3_4B_INSTRUCT_2507_DEFAULT_ARGS = {
    "model": "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
    "max_model_len": 8192,
    "max_completion_tokens": 4096,
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 1,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
}

QWEN3_30B_A3B_Thinking_2507_FP8_DEFAULT_ARGS = {
    "model": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "max_model_len": 81_920,
    "max_completion_tokens": 61_440,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}

GPT_OSS_120B_HIGH_DEFAULT_ARGS = {
    "model": "openai/gpt-oss-120b",
    "max_model_len": 81_920, # the model reasons even 12.5k samples
    "max_completion_tokens": 61_440,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.9, # recommended is 1 but from my tests 0.9 is better (1 fails a little in the final answer)
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "high",
    "ignore_patterns": ["original/**", "metal/**"]
}

GPT_OSS_120B_MEDIUM_DEFAULT_ARGS = {
    "model": "openai/gpt-oss-120b",
    "max_model_len": 81_920, # the model reasons even 12.5k samples
    "max_completion_tokens": 61_440,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.9, # recommended is 1 but from my tests 0.9 is better (1 fails a little in the final answer)
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "medium",
    "ignore_patterns": ["original/**", "metal/**"]
}

GPT_OSS_120B_LOW_DEFAULT_ARGS = {
    "model": "openai/gpt-oss-120b",
    "max_model_len": 81_920, # the model reasons even 12.5k samples
    "max_completion_tokens": 61_440,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.9, # recommended is 1 but from my tests 0.9 is better (1 fails a little in the final answer)
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "low",
    "ignore_patterns": ["original/**", "metal/**"]
}


GPT_OSS_20B_LOW_DEFAULT_ARGS = {
    "model": "openai/gpt-oss-20b",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.9,
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "low",
    "ignore_patterns": ["original/**", "metal/**"]
}

GPT_OSS_20B_HIGH_DEFAULT_ARGS = {
    "model": "openai/gpt-oss-20b",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.9,
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "high",
    "ignore_patterns": ["original/**", "metal/**"]
}

MISTRAL_SMALL_2506_DEFAULT_ARGS = {
    "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.15,
    "top_p": 1,
    "reasoning_effort": "high",
    "ignore_patterns": ["original/**", "metal/**"],
    "other_args": ["--tokenizer-mode", "mistral","--config_format", "mistral", "--load_format", "mistral"]
}

MAGISTRAL_SMALL_2509_DEFAULT_ARGS = {
    "model": "mistralai/Magistral-Small-2509",
    "max_model_len": 40_960,
    "max_completion_tokens":30_720,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.7, # recommmended by them
    "top_p": 0.95, # recommended by them
    "reasoning_effort": "high",
    "ignore_patterns": ["original/**", "metal/**"],
    "other_args": [ "--tokenizer-mode", "mistral","--config_format", "mistral", "--load_format", "mistral"]
}

GEMMA3_27B_IT_DEFAULT_ARGS = {
    "model": "unsloth/gemma-3-27b-it-bnb-4bit",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "temperature": 0.7,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "top_k": 64,
    "top_p": 0.95
}