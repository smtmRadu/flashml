
from typing import Literal
def vllm_chat(
    model_name:str,
    tokenizer_name:str,
    quantization:Literal["awq", "gptq", "awq_marlin"],
    messages:list[dict] | list[list[dict]],
    
    max_model_len:int= 4096,
    batch_size=64,
    gpu_memory_utilization=0.9,
    # sampling
    temperature:float=1,
    top_k=-1, 
    top_p=1,
    max_tokens=131_072,
    format:dict[str, any]=None
):
    """
    (WSL or Linux only) Runs chat inference on a VLLM backend.
    Install FlashInfer for best performance.

    Args:
        model_name (str): The name or path of the model to use.
        tokenizer_name (str): The name or path of the tokenizer to use.
        quantization (Literal["awq", "gptq", "awq_marlin"]): 
            The quantization format to load the model weights with.
        messages (list[dict] | list[list[dict]]): 
            Chat history in OpenAI format (single or batch). Each message should be a dict
            with fields like 'role' and 'content'.
        max_model_len (int, optional): 
            Maximum number of tokens per chat. VLLM backend prefills the VRAM memory for KV caches so it must know your limit. Defaults to 4096.
        batch_size (int, optional): 
            Increase this as much as you can until the VLLM fails to initialize. Decrease if VLLM fails to initialize. Defaults to 64.
        temperature (float, optional): 
            Sampling temperature. Higher values produce more random outputs. Defaults to 1.
        top_k (int, optional): 
            Number of top tokens to sample from. Set to -1 to disable top-k sampling. Defaults to -1.
        top_p (float, optional): 
            Cumulative probability for nucleus sampling. Defaults to 1 (no filtering).
        max_tokens (int, optional): 
            Maximum number of tokens to generate in the output. Defaults to 65,536.
        format (dict[str, any], optional): 
            A model_json_schema() from a pydantic BaseModel class.

    Returns:
        list[dict]: The generated responses for each conversation in the input, formatted as OpenAI-compatible message dicts.

    Raises:
        ValueError: If the provided quantization type is not supported.
        RuntimeError: If model loading or inference fails.

    Example:
        ```python
        messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are Qwen."},
            {"role": "user", "content": "Hi there!"},
        ]] # two parallel conversations
        
        response = vllm_chat(
            model_name="Qwen/Qwen3-4B-AWQ",
            tokenizer_name="Qwen/Qwen3-4B",
            quantization="awq_marlin",
            messages=messages
        )
        
        for i in response:
            print(i.outputs[0].text)
        ```
    """  
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    if VLLMCore.llm is None:
        VLLMCore.llm = LLM(
        model= model_name,
        tokenizer=tokenizer_name,
        quantization=quantization,
        max_model_len=max_model_len,
        max_num_seqs= batch_size,
        gpu_memory_utilization = gpu_memory_utilization
    )
        
    return VLLMCore.llm.chat(
        messages=messages,
        sampling_params=SamplingParams(max_tokens=max_tokens,temperature=temperature, top_k=top_k, top_p=top_p, guided_decoding=GuidedDecodingParams(json=format)),
        use_tqdm=True
    )

class VLLMCore():
    llm = None
        

