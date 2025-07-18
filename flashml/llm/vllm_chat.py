
from typing import Literal
def vllm_chat(
    model_name:str,
    tokenizer_name:str,
    quantization:Literal["awq", "gptq", "awq_marlin"],
    messages:list[dict] | list[list[dict]],
    
    max_model_len:int= 4096,
    batch_size=64,
    # sampling
    temperature:float=1,
    top_k=-1, 
    top_p=1,
    max_tokens=65_536,
    format:dict[str, any]=None
):
    """
    
    Note that this works in Linux and WSL only
    model_name:str Quantized model name.
    tokenizer_name:str Base model name (tokenizer is extracted from the base model)
    quantization:str The quantization of the model (is in name)
    messages : the full batch of messages
    
    Extract the content of each elem in response like this:
        output[__index__].outputs[0].text
    """
    
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    if VLLMCore.llm is None:
        VLLMCore.llm = LLM(
        model= model_name,
        tokenizer=tokenizer_name,
        quantization=quantization,
        max_model_len=max_model_len,
        max_num_seqs= batch_size
    )
        
    return VLLMCore.llm.chat(
        messages=messages,
        sampling_params=SamplingParams(max_tokens=max_tokens,temperature=temperature, top_k=top_k, top_p=top_p, guided_decoding=GuidedDecodingParams(json=format)),
        use_tqdm=True
    )

class VLLMCore():
    llm = None
        

