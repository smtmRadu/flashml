from .plot_chat import plot_chat
from .openai_api_batch_builder import OpenAIBatchRequest, OpenAISyncRequest
from .vllm_chat import vllm_chat
from .vllm_compute_logprobs import vllm_compute_logprobs
from .vllm_engine import vllm_close
from .vllm_chat_openai_entrypoint import vllm_chat_openai_entrypoint

from .llm_merging import  merge_unsloth_model
from .llm_quantization import quantize_model
from .llm_utils import (
get_bnb_4bit_quantization_config, 
get_boxed_answer, 
image_to_base64)

# there are deprecated
#merge_and_quantize_llm, 
#quantize_llm,
# merge_llm)


from .vllm_configs import (
    QWEN3_0_6B_CONFIG_VLLM_CONFIG, 
    QWEN3_4B_THINKING_2507_VLLM_CONFIG,
    QWEN3_VL_2B_THINKING_VLLM_CONFIG,
    QWEN3_VL_4B_THINKING_AWQ_VLLM_CONFIG,
    QWEN3_VL_8B_THINKING_VLLM_CONFIG,
    QWEN3_VL_30B_A3B_THINKING_VLLM_CONFIG,
    GEMMA3_4B_IT_VLLM_CONFIG,
    GEMMA3_27B_IT_VLLM_CONFIG,
    MINISTRAL_3_3B_INSTRUCT_2512_VLLM_CONFIG,
    
    GPT_OSS_20B_LOW_VLLM_CONFIG, 
    GPT_OSS_120B_LOW_VLLM_CONFIG,
    GPT_OSS_120B_HIGH_VLLM_CONFIG)

__all__ = [
    
    "GPT_OSS_20B_LOW_VLLM_CONFIG",
    "GPT_OSS_120B_HIGH_VLLM_CONFIG",
    "GPT_OSS_120B_LOW_VLLM_CONFIG",
    
    "QWEN3_0_6B_CONFIG_VLLM_CONFIG",
    "QWEN3_4B_THINKING_2507_VLLM_CONFIG",
    "QWEN3_VL_2B_THINKING_VLLM_CONFIG",
    "QWEN3_VL_4B_THINKING_AWQ_VLLM_CONFIG",
    "QWEN3_VL_8B_THINKING_VLLM_CONFIG",
    "QWEN3_VL_30B_A3B_THINKING_VLLM_CONFIG",
    
    "MINISTRAL_3_3B_INSTRUCT_2512_VLLM_CONFIG",
    
    "GEMMA3_4B_IT_VLLM_CONFIG",
    "GEMMA3_27B_IT_VLLM_CONFIG",
    
    "OpenAIBatchRequest",
    "OpenAISyncRequest",
    "get_bnb_4bit_quantization_config",
    "get_boxed_answer",
    "image_to_base64",
    
    #"merge_llm",
    #"quantize_llm",
    #"merge_and_quantize_llm",
    
    "quantize_model",
    "merge_unsloth_model",
    
    "vllm_chat_openai_entrypoint",
    "plot_chat",
    "vllm_chat",
    "vllm_close",
    "vllm_compute_logprobs",
]