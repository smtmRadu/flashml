from typing import Literal
import os 
import shutil
import json
GREEN = '\033[32m'
RESET = '\033[0m'
BLUE = '\033[34m'
RED = "\033[31m"


def quantize_model(model_path, quantization:Literal["bnb_4bit"]="bnb_4bit"):
    """
    Loads an fp16 model that was previously merged and quantizes it.
    """
    
    # raise "if using just AutoModel instead of AutoModelForCausalLM i get this error: AttributeError: 'Gemma3Config' object has no attribute 'vocab_size'"
    # remember to copy paste the preprocessor
    if not os.path.exists(model_path,):
        raise ValueError(f"Adapter path {model_path} does not exist.")
    
    
    with open(model_path + "/config.json", "r") as file:
        config_data = json.load(file)
        # print(config_data)
        if config_data["model_type"].lower() == "mistral3":
# 
            print(f"{BLUE}[INFO] {RED}Mistral model detected.")
            _quantize_mistral_model(model_path, quantization)
            return
            
    
    
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
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        quantized_path = model_path.replace("_fp16", "_bnb_4bit")
        print(f"{BLUE}[Step 2] Saving to {GREEN}{quantized_path}{RESET} ...")
        model.save_pretrained(quantized_path, safe_serialization=True)
        tokenizer.save_pretrained(quantized_path)
    else:   
        raise Exception(f"Unhandled quantization method {quantization}")
    print("✅ Quantization complete!")
    
    
def _quantize_mistral_model(model_path, quantization:Literal["bnb_4bit"]="bnb_4bit"):
    
    """
    Loads an fp16 model that was previously merged and quantizes it.
    """
    # raise "Not fucking works. It seems like automodel loads a 10/10 safetensors mistral idk how."
    # raise "if using just AutoModel instead of AutoModelForCausalLM i get this error: AttributeError: 'Gemma3Config' object has no attribute 'vocab_size'"
    # remember to copy paste the preprocessor
    if not os.path.exists(model_path,):
        raise ValueError(f"Adapter path {model_path} does not exist.")
    
    from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
    from transformers import BitsAndBytesConfig
    import torch
    if quantization == "bnb_4bit":
        print(f"{BLUE}[Step 1] Loading {RED}(model) {model_path}{RESET} ...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            offload_folder="./offload_flashml",
            quantization_config=bnb_config,
            trust_remote_code=True,
            )
        
        quantized_path = model_path.replace("_fp16", "_bnb_4bit")
        print(f"{BLUE}[Step 2] Saving to {GREEN}{quantized_path}{RESET} ...")
        model.save_pretrained(quantized_path, safe_serialization=True)
        
        ## copy tokenizer and tokenizer_config files to the new folder 
        
        tokenizer_path = model_path + "/tokenizer.json"
        tokenizer_config_path = model_path + "/tokenizer_config.json"
        new_folder = quantized_path
        if os.path.exists(tokenizer_path):
            shutil.copy(tokenizer_path, new_folder + "/tokenizer.json")
        if os.path.exists(tokenizer_config_path):
            shutil.copy(tokenizer_config_path, new_folder + "/tokenizer_config.json")
        
        
    else:   
        raise Exception(f"Unhandled quantization method {quantization}")
    print("✅ Quantization complete!")
