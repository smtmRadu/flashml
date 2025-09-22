from flashml.llm.vllm_engine import VLLMCore
from typing import Literal, List
def vllm_chat(
    messages:list[dict] | list[list[dict]],
    model_name:str,
    tokenizer_name:str = None,
    tokenizer_mode:str = 'auto',
    enforce_eager:bool|None=None,
    disable_custom_all_reduce:bool=False,
    quantization:Literal["awq", "gptq", "awq_marlin", 'gptq_marlin', "bitsandbytes"] = None,
    max_model_len:int= 4096,
    max_num_seqs:int=256,
    tensor_parallel_size:int=1,
    gpu_memory_utilization=0.85,
    
    # sampling
    temperature:float=1,
    top_k=-1, 
    top_p=1,
    min_p=0,
    stop: List[str] | List[int] = None, 
    format:dict[str, any]=None,
    max_tokens=131_072,
    ignore_patterns=["original/**", "metal/**", "consolidated.safetensors"],
    **kwargs,
):
    """
    (WSL or Linux only) Runs chat inference on a VLLM backend.
    It handles None messages (output will be None as well)
    Install FlashInfer for best performance.

    Args:
        model_name (str): The name or path (if local) of the model to use.
        tokenizer_name (str): The name or path of the tokenizer to use. If None, it is set as the model_name,
        quantization (Literal["awq", "gptq", "awq_marlin"]): 
            The quantization format to load the model weights with.
        messages (list[dict] | list[list[dict]]): 
            Chat history in OpenAI format (single or batch). Each message should be a dict
            with fields like 'role' and 'content'. May contain None values.. the model with not answer them.
        max_model_len (int, optional): 
            Maximum number of tokens per chat. VLLM backend prefills the VRAM memory for KV caches so it must know your limit. Defaults to 4096.
        max_num_seqs(int):
            Decrease this if the engine fails to init.
        temperature (float, optional): 
            Sampling temperature. Higher values produce more random outputs. Defaults to 1.
        top_k (int, optional): 
            Number of top tokens to sample from. Set to -1 to disable top-k sampling. Defaults to -1.
        top_p (float, optional): 
            Cumulative probability for nucleus sampling. Defaults to 1 (no filtering).
        stop (list[int], list[str]):
            The strings the model will stop from generating or the indices of the tokens to stop from generating. These are not included in the output.
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
            model_name="warshanks/Qwen3-4B-Thinking-2507-AWQ",
            tokenizer_name="Qwen/Qwen3-4B",
            quantization="awq_marlin",
            messages=messages
        )
        
        for i in response:
            print(i.outputs[0].text)
        ```
    """  
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = VLLMCore.initialize(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        tokenizer_mode=tokenizer_mode,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=disable_custom_all_reduce,
        quantization=quantization,
        max_model_len=max_model_len, 
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        ignore_patterns=ignore_patterns,
        **kwargs)
        
    if isinstance(messages, list) and all(isinstance(i, list) or i == None for i in messages):
        non_none_messages = [m for m in messages if m is not None]
        
        non_none_outputs = llm.chat(
            messages=non_none_messages,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k, 
                top_p=top_p,
                min_p=min_p, 
                
                stop=stop if stop is not None and isinstance(stop[0], str) else None,
                stop_token_ids=stop if stop is not None and isinstance(stop[0], int) else None,
                guided_decoding=GuidedDecodingParams(json=format) if format is not None else None),
            use_tqdm=True
        )
        
        with_none_outputs = []
        index_in_outp = 0
        for  m in messages:
            if m is None:
                with_none_outputs.append(None)
            else:
                with_none_outputs.append(non_none_outputs[index_in_outp])
                index_in_outp += 1
                
        return with_none_outputs
    else:
        return llm.chat(
            messages=messages,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature, 
                top_k=top_k,
                top_p=top_p, 
                min_p=min_p, 
                stop=stop if stop is not None and isinstance(stop[0], str) else None,
                stop_token_ids=stop if stop is not None and isinstance(stop[0], int) else None,
                guided_decoding=GuidedDecodingParams(json=format) if format is not None else None),
            use_tqdm=True
        )

        

