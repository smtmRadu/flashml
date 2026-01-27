from typing import Literal
import os 
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
    else:   
        raise Exception(f"Unhandled quantization method {quantization}")
    print("✅ Quantization complete!")