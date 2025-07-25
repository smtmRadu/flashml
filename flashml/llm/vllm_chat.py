from flashml.llm.vllm_engine import VLLMCore
from typing import Literal
def vllm_chat(
    messages:list[dict] | list[list[dict]],
    model_name:str,
    tokenizer_name:str,
    quantization:Literal["awq", "gptq", "awq_marlin"],
    max_model_len:int= 4096,
    max_num_seqs=256,
    gpu_memory_utilization=0.8,
    # sampling
    temperature:float=1,
    top_k=-1, 
    top_p=1,
    format:dict[str, any]=None,
    max_tokens=131_072,
):
    """
    (WSL or Linux only) Runs chat inference on a VLLM backend.
    It handles None messages (output will be None as well)
    Install FlashInfer for best performance.

    Args:
        model_name (str): The name or path of the model to use.
        tokenizer_name (str): The name or path of the tokenizer to use.
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
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = VLLMCore.initialize(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        quantization=quantization,
        max_model_len=max_model_len, 
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization)
        
    if isinstance(messages, list) and all(isinstance(i, list) for i in messages):
        non_none_messages = [m for m in messages if m is not None]
        
        non_none_outputs = llm.chat(
            messages=non_none_messages,
            sampling_params=SamplingParams(max_tokens=max_tokens,temperature=temperature, top_k=top_k, top_p=top_p, guided_decoding=GuidedDecodingParams(json=format) if format is not None else None),
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
            sampling_params=SamplingParams(max_tokens=max_tokens,temperature=temperature, top_k=top_k, top_p=top_p, guided_decoding=GuidedDecodingParams(json=format) if format is not None else None),
            use_tqdm=True
        )

        

