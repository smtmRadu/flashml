from typing import Literal, List
from flashml.llm.vllm_engine import VLLMCore

def vllm_compute_logprobs(
    texts: List[str],
    model_name: str,
    tokenizer_name: str = None,
    tokenizer_mode:str = 'auto',
    enforce_eager:bool|None=False,
    disable_custom_all_reduce:bool=False,
    quantization: Literal["awq", "gptq", "awq_marlin"] = None,
    max_model_len: int = 32_768,
    max_num_seqs: int = 256,
    tensor_parallel_size=1,
    gpu_memory_utilization: float = 0.8,
    ignore_patterns=["original/**", "metal/**"]
) -> List[List[float]]:
    """
    Gets logprobs of the actual tokens of the text using VLLM and a local LLM.
    
    Args:
        model_name (str): The name or path of the model to use.
        tokenizer_name (str): The name or path of the tokenizer to use.
        texts (List[str]): List of input texts to analyze.
        quantization (Literal["awq", "gptq", "awq_marlin"]): 
            The quantization format to load the model weights with.
        max_model_len (int): Maximum context length.
        max_num_seqs (int): Maximum number of sequences to process in parallel.
        gpu_memory_utilization (float): GPU memory utilization ratio.
        return_with_tokens (bool): if False, returns only the logprobs. If True, returns both logprobs and tokens.
    Returns:
        List[List[float]]: List of log probability lists, one per input text.
                          Each inner list contains the log probability of each token in that text.
    
    Example:
        ```python
        texts = ["Hello world", "The year is 2024"]
        results = vllm_get_token_logprobs(
            model_name="Qwen/Qwen3-4B-AWQ",
            tokenizer_name="Qwen/Qwen3-4B", 
            texts=texts
        )
        
        for i, logprobs in enumerate(results):
            print(f"Text {i}: '{texts[i]}'")
            print(f"Logprobs: {logprobs}")
        ```
    """
    from vllm import SamplingParams
    
    # Initialize model if not already loaded
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
        ignore_patterns=ignore_patterns)
    
    # Set up sampling params to get logprobs
    # We generate 0 tokens but request logprobs for the input tokens
    sampling_params = SamplingParams(
        max_tokens=1,  # Don't generate new tokens
        logprobs=1,  # We only need the actual token's logprob
        prompt_logprobs=1  # Get logprobs for input tokens
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name is not None else model_name)
    
    args = {
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "quantization": quantization,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "ignore_patterns": ignore_patterns
        }
    
    if isinstance(texts, list) and all(isinstance(i, list) or i is None for i in texts):
        non_none_texts = [x for x in texts if x is not None]
        
        # Process texts in batch
        outputs = llm.generate(
            prompts=non_none_texts,
            sampling_params=sampling_params,
            use_tqdm=True
        )
        
        full_info = []

        output_idx = 0
        for t in texts:
            if t is None:
                full_info.append(None)
            else:
                output = outputs[output_idx]
                output_idx += 1
                input_ids = output.prompt_token_ids
                token_logprobs = [None]  # No logprob for the first token
                tokens = [tokenizer.decode(output.prompt_token_ids[0])]
                ranks = [None]
                for i in range(1, len(input_ids)):
                    logprob_dict = output.prompt_logprobs[i]
                    token_id = input_ids[i]
                    key = token_id if token_id in logprob_dict else str(token_id)
                    logprob_obj = logprob_dict.get(key)

                    token_logprobs.append(logprob_obj.logprob)
                    tokens.append(logprob_obj.decoded_token)
                    ranks.append(logprob_obj.rank)
                    
                full_info.append({
                    "request_id": output.request_id,
                    "prompt": output.prompt,
                    "logprobs": token_logprobs,
                    "tokens_ids": output.prompt_token_ids,
                    "tokens": tokens,
                    "ranks": ranks,
                    "args": args
                })
                
        
        

        return full_info


        
    else:
        full_info = []
        outputs = llm.generate(
            prompts=texts,
            sampling_params=sampling_params,
            use_tqdm=True
        )
        
        for output in outputs:
            input_ids = output.prompt_token_ids
            token_logprobs = [None]
            tokens = [tokenizer.decode(output.prompt_token_ids[0])]
            ranks = [None]
            for i in range(1, len(input_ids)):
                logprob_dict = output.prompt_logprobs[i]
                token_id = input_ids[i]
                key = token_id if token_id in logprob_dict else str(token_id)
                logprob_obj = logprob_dict.get(key)
                token_logprobs.append(logprob_obj.logprob)
                tokens.append(logprob_obj.decoded_token)
                ranks.append(logprob_obj.rank)
                    
            full_info.append({
                    "request_id": output.request_id,
                    "prompt": output.prompt,
                    "logprobs": token_logprobs,
                    "tokens_ids": output.prompt_token_ids,
                    "tokens": tokens,
                    "ranks": ranks,
                    "args": args
                })
        
        
        
        return full_info
