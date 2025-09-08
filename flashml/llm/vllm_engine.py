from typing import Literal

def vllm_close():
    import gc
    '''
    Stops the vllm instance.
    '''
    del VLLMCore._instance
    VLLMCore._instance = None
    gc.collect()
    
class VLLMCore():
    _instance = None
    _instance_current_args = None
    def initialize(
        model_name:str, 
        tokenizer_name:str, 
        quantization:Literal["awq", "gptq", "awq_marlin", "gptq_marlin", "bitsandbytes"],
        max_model_len:int= 4096,
        max_num_seqs=256,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        ignore_patterns=None):
            """Returns a vllm instance LLM class.

            Args:
                model_name (str): _description_
                tokenizer_name (str): _description_
                quantization (Literal[&quot;awq&quot;, &quot;gptq&quot;, &quot;awq_marlin&quot;]): _description_
                max_model_len (int, optional): _description_. Defaults to 4096.
                max_num_seqs (int, optional): _description_. Defaults to 256.
                tensor_parallel_size (int, optional): _description_. Defaults to 1.
                gpu_memory_utilization (float, optional): _description_. Defaults to 0.9.

            Returns:
                _type_: _description_
            """
            if tokenizer_name is None:
                tokenizer_name = model_name
            from vllm import LLM
            if VLLMCore._instance is None:
                VLLMCore._instance = LLM(
                    model= model_name,
                    tokenizer=tokenizer_name,
                    quantization=quantization,
                    max_model_len=max_model_len,
                    max_num_seqs= max_num_seqs,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization = gpu_memory_utilization,
                    ignore_patterns=ignore_patterns,
                )
                VLLMCore._instance_current_args = {
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                    "quantization": quantization,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "ignore_patterns": ignore_patterns
                }
                return VLLMCore._instance
            
            # if similar init args then return the _instance else reinit core
            
            similar_core = True
            if VLLMCore._instance is not None:
                if VLLMCore._instance_current_args["model_name"] != model_name:
                    similar_core = False
                if VLLMCore._instance_current_args["tokenizer_name"]  != tokenizer_name:
                    similar_core = False
                if VLLMCore._instance_current_args["quantization"]!= quantization:
                    similar_core = False   
                if VLLMCore._instance_current_args["max_model_len"]!= max_model_len:
                    similar_core = False
                if VLLMCore._instance_current_args["max_num_seqs"]!= max_num_seqs:
                    similar_core = False
                if VLLMCore._instance_current_args["gpu_memory_utilization"]!= gpu_memory_utilization:
                    similar_core = False
                if VLLMCore._instance_current_args["tensor_parallel_size"]!= tensor_parallel_size:
                    similar_core = False
                if VLLMCore._instance_current_args["ignore_patterns"]!= ignore_patterns:
                    similar_core = False
                    
            if similar_core:
                return VLLMCore._instance
            
            # VLLMCore._instance... it closes automatically.
            del VLLMCore._instance
            VLLMCore._instance = None
            import gc
            gc.collect()
            VLLMCore._instance = LLM(
                    model= model_name,
                    tokenizer=tokenizer_name,
                    quantization=quantization,
                    max_model_len=max_model_len,
                    max_num_seqs= max_num_seqs,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization = gpu_memory_utilization,
                    ignore_patterns=ignore_patterns,
                )
            VLLMCore._instance_current_args = {
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                    "quantization": quantization,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "ignore_patterns": ignore_patterns
                }
            
            return VLLMCore._instance   