from typing import Literal, List

def vllm_get_token_logprobs(
    model_name: str,
    tokenizer_name: str,
    quantization: Literal["awq", "gptq", "awq_marlin"],
    texts: List[str],
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    gpu_memory_utilization: float = 0.9,
) -> List[List[float]]:
    """
    Gets logprobs of the actual tokens that appear in each text using VLLM.
    
    Args:
        model_name (str): The name or path of the model to use.
        tokenizer_name (str): The name or path of the tokenizer to use.
        texts (List[str]): List of input texts to analyze.
        quantization (Literal["awq", "gptq", "awq_marlin"]): 
            The quantization format to load the model weights with.
        max_model_len (int): Maximum context length.
        max_num_seqs (int): Maximum number of sequences to process in parallel.
        gpu_memory_utilization (float): GPU memory utilization ratio.
    
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
    from vllm import LLM, SamplingParams
    
    # Initialize model if not already loaded
    if VLLMCore.llm is None:
        VLLMCore.llm = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            quantization=quantization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization
        )
    
    # Get tokenizer for processing
    tokenizer = VLLMCore.llm.get_tokenizer()
    
    # Set up sampling params to get logprobs
    # We generate 0 tokens but request logprobs for the input tokens
    sampling_params = SamplingParams(
        max_tokens=1,  # Don't generate new tokens
        logprobs=1,  # We only need the actual token's logprob
        prompt_logprobs=True  # Get logprobs for input tokens
    )
    
    results = []
    
    # Process texts in batch
    outputs = VLLMCore.llm.generate(
        prompts=texts,
        sampling_params=sampling_params,
        use_tqdm=True
    )
    
    for i, output in enumerate(outputs):
        text = texts[i]
        
        # Get tokens and token IDs
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        
        # Extract logprobs from output - only the logprob of the actual token
        token_logprobs = [0.0]
        for j, logprob_dict in enumerate(output.prompt_logprobs[1:]):
            idx = j + 1
            if logprob_dict is not None and idx < len(input_ids):
                actual_token_id = input_ids[idx]
                key = actual_token_id
                if key not in logprob_dict and isinstance(key, int):
                    key = str(key)
                if key in logprob_dict:
                    token_logprobs.append(logprob_dict[key].logprob)
                else:
                    token_logprobs.append(0.0)
            else:
                token_logprobs.append(0.0)
        
        # If we don't have enough logprobs, pad with 0.0
        while len(token_logprobs) < len(input_ids):
            token_logprobs.append(0.0)
        
        # Trim if we have too many
        token_logprobs = token_logprobs[:len(input_ids)]
        
        results.append(token_logprobs)
    
    return results


class VLLMCore:
    llm = None