from typing import Literal
import os 
import json
import shutil
GREEN = '\033[32m'
RESET = '\033[0m'
BLUE = '\033[34m'
RED = "\033[31m"
def quantize_model(model_path, quantization:Literal["bnb_4bit"]="bnb_4bit"):
    # raise "if using just AutoModel instead of AutoModelForCausalLM i get this error: AttributeError: 'Gemma3Config' object has no attribute 'vocab_size'"
    # remember to copy paste the preprocessor
    if not os.path.exists(model_path,):
        raise ValueError(f"Adapter path {model_path,} does not exist.")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    import torch
    if quantization == "bnb_4bit":
        print(f"{BLUE}[Step 1] Loading {RED}(model) {model_path}{RESET} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            device_map="auto",
            dtype=None,
            offload_folder="./offload_flashml",
            trust_remote_code=True, 
            load_in_4bit=False,        
            load_in_8bit=False 
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"{BLUE}[Step 2] Saving to {GREEN}{model_path.replace("_fp16", "_bnb_4bit")}{RESET} ...")
        model.save_pretrained(model_path.replace("_fp16", "_bnb_4bit"), safe_serialization=True)
        tokenizer.save_pretrained(model_path.replace("_fp16", "_bnb_4bit"))
    
    print("✅ Quantization complete!")
    
def merge_unsloth_model(adapter_path: str, save_method:Literal["fp16", "gguf"] = "fp16"):
    """
    When training with unsloth, merge the adapter to the base model using this function and save it to fp16.
    You can quantize it afterwards and save it, or quantize it at runtime.
    
    The current function is still strange, it copies the hf cache to the save folder and takes a while, but anyways...
    """
    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter path {adapter_path} does not exist.")
    
    config_path = adapter_path + "/adapter_config.json"
    if not os.path.exists(config_path):
        raise ValueError(f"adapter_config.json not found in {adapter_path}.")
    
    with open(config_path) as f:
        adapter_config = json.load(f)    
    
    base_model_path_or_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path_or_path.startswith("unsloth"):
        raise ValueError("This is not an unsloth model. Please use the normal `merge_model` function to merge in transformers.")
    ## Just a little warning
    if "gemma" in base_model_path_or_path and not os.path.exists(adapter_path + "/preprocessor_config.json"):
        raise Exception("⚠️ `preprocessor_config.json` and `processor_config.json` files are missing for gemma adapter. Please download them from https://huggingface.co/google/gemma-3-4b-it/tree/main and add them manually to the adapter.")
    
    
    was_4bit_training = "4bit" in base_model_path_or_path   
    print(f"{BLUE}[Step 1] Loading {RED}(model) {base_model_path_or_path}{RESET} + {GREEN}(adapter) {adapter_path}{RESET}...")
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name =  adapter_path, 
        max_seq_length = 2048, 
        device_map="auto",
        offload_folder="./offload_flashml",
        # local_files_only = True,  # Force local cache usage
        dtype = None,          
        load_in_4bit = False, # was_4bit_training
    )

    if save_method == "bnb_4bit": # I tested it out, it merges it but vllm doesn't manage to load it. I recommend quantize it at runtime (add --quantization bitsandbytes in vllm instantiation)
        if "gemma" in base_model_path_or_path:
            raise ValueError("We do not recommend saving in 4bit the gemma models. vLLM fails to load it due some parameter shape difference. Save it in fp16")
        merge_name = adapter_path + "_bnb_4bit"
        print(f"{BLUE}[Step 2] Saving {GREEN}{merge_name}{RESET}... (Merging into 4bit will cause your model to lose accuracy if you planto merge to GGUF or others later on. I suggest you to do this as a final step if you're planning to do multiple saves.)")
        model.save_pretrained_merged(
            merge_name, 
            tokenizer,
            save_method = "merged_4bit_forced"
        )
    elif save_method == "fp16":
        merge_name = adapter_path + "_fp16"
        print(f"{BLUE}[Step 2] Saving {GREEN}{merge_name}{RESET} ...")
        model.save_pretrained_merged(
            merge_name, 
            tokenizer,
            save_method = "merged_16bit"
        )
    elif save_method == "gguf":
        merge_name = adapter_path + "_gguf_Q8_0"
        print(f"{BLUE}[Step 2] Saving {GREEN}{merge_name}{RESET} ...")
        model.save_pretrained_gguf(
            merge_name,  
            tokenizer,
            quantization_method = "Q8_0", # Only Q8_0, BF16, F16 supported
        )
    if os.path.exists("./offload_flashml"):
        shutil.rmtree("./offload_flashml")
    # save_method = "lora" saves only the adapter
    
    
    print("✅ Merge complete!")