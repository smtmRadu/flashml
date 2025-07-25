from typing import Literal

class VLLMCore():
    _instance = None
    _instance_current_args = None
    def initialize(
        model_name:str, 
        tokenizer_name:str, 
        quantization:Literal["awq", "gptq", "awq_marlin"],
        max_model_len:int= 4096,
        max_num_seqs=256,
        gpu_memory_utilization=0.8):
            """Returns a vllm instance LLM class.

            Args:
                model_name (str): _description_
                tokenizer_name (str): _description_
                quantization (Literal[&quot;awq&quot;, &quot;gptq&quot;, &quot;awq_marlin&quot;]): _description_
                max_model_len (int, optional): _description_. Defaults to 4096.
                max_num_seqs (int, optional): _description_. Defaults to 256.
                gpu_memory_utilization (float, optional): _description_. Defaults to 0.9.

            Returns:
                _type_: _description_
            """
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import GuidedDecodingParams
            
            if VLLMCore._instance is None:
                VLLMCore._instance = LLM(
                    model= model_name,
                    tokenizer=tokenizer_name,
                    quantization=quantization,
                    max_model_len=max_model_len,
                    max_num_seqs= max_num_seqs,
                    gpu_memory_utilization = gpu_memory_utilization
                )
                VLLMCore._instance_current_args = {
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                    "quantization": quantization,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "gpu_memory_utilization": gpu_memory_utilization
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
                
            if similar_core:
                return VLLMCore._instance
            
            # VLLMCore._instance... it closes automatically.
            VLLMCore._instance = LLM(
                    model= model_name,
                    tokenizer=tokenizer_name,
                    quantization=quantization,
                    max_model_len=max_model_len,
                    max_num_seqs= max_num_seqs,
                    gpu_memory_utilization = gpu_memory_utilization
                )
            VLLMCore._instance_current_args = {
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                    "quantization": quantization,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "gpu_memory_utilization": gpu_memory_utilization
                }
            
            return VLLMCore._instance   