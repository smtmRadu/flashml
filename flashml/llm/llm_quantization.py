from typing import Literal
import os 
import shutil
import json
from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse as parse_version
GREEN = '\033[32m'
RESET = '\033[0m'
BLUE = '\033[34m'
RED = "\033[31m"


def _package_version_or_none(package_name: str):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _uses_cpu_only(device_map) -> bool:
    if device_map == "cpu":
        return True
    if isinstance(device_map, dict):
        return all(str(value).lower() == "cpu" for value in device_map.values())
    return False


def _ensure_gptq_backend_available(device_map) -> None:
    optimum_version = _package_version_or_none("optimum")
    gptqmodel_version = _package_version_or_none("gptqmodel")
    auto_gptq_version = _package_version_or_none("auto-gptq")

    if optimum_version is None:
        raise ImportError(
            "GPTQ quantization requires `optimum`. Install it with "
            "`python -m pip install 'optimum>=1.24.0'`."
        )

    if gptqmodel_version is None and auto_gptq_version is None:
        raise ImportError(
            "GPTQ quantization requires `gptqmodel` or `auto-gptq`, but neither is installed. "
            "Install a backend first, for example "
            "`python -m pip install 'gptqmodel>=1.4.3' 'transformers>=4.57.1,<5'`."
        )

    if gptqmodel_version is not None and parse_version(optimum_version) < parse_version("1.24.0"):
        raise ImportError(
            f"`gptqmodel` requires `optimum>=1.24.0`, found {optimum_version}."
        )

    if (
        _uses_cpu_only(device_map)
        and gptqmodel_version is None
        and auto_gptq_version is not None
        and parse_version(auto_gptq_version) < parse_version("0.4.2")
    ):
        raise ImportError(
            f"CPU GPTQ quantization requires `gptqmodel` or `auto-gptq>=0.4.2`, found auto-gptq "
            f"{auto_gptq_version}."
        )


def quantize_model(model_path, device_map="auto", quantization:Literal["bnb_4bit", "gptq_2bit", "gptq_3bit","gptq_4bit","gptq_8bit"]="bnb_4bit", calibration_set:list[str]=None):
    """
    Loads an fp16 model that was previously merged and quantizes it.
    
    For calibration, I recommend applying chat template over the entire conversation and pass it to the calibration set list.
    256 samples are enough, use the following guiding script
    ```
    MODEL = ...
    dataset = pl.read_csv(...)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    def format_sample(system, user, assistant):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    calibration_set = [format_sample(x["system"], x["user"], x["assistant"]) for x in dataset.sample(256).iter_rows(named=True)]
    ```
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
            _quantize_mistral_model(model_path, device_map, quantization, calibration_set)
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
            device_map=device_map,
            dtype=None,
            offload_folder="./offload_flashml",
            trust_remote_code=True, 
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        quantized_path = model_path.replace("_fp16", "_bnb_4bit")
        print(f"{BLUE}[Step 2] Saving to {GREEN}{quantized_path}{RESET} ...")
        model.save_pretrained(quantized_path, safe_serialization=True)
        tokenizer.save_pretrained(quantized_path)
    elif quantization.startswith("gptq_"):
        if calibration_set is None:
            raise ValueError("GPTQ quantization requires a calibration dataset")
        _ensure_gptq_backend_available(device_map)

        from transformers import AutoTokenizer, GPTQConfig
        from transformers import AutoModelForCausalLM

        print(f"{BLUE}[Step 1] Loading tokenizer from {RED}{model_path}{RESET} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # calibration_set should be a list of strings, e.g. ["Hello world", "The cat sat..."]
        gptq_config = GPTQConfig(
            bits=int(quantization[5]),                        # 2, 3, 4, or 8
            dataset=calibration_set,       # list[str] or a dataset name like "c4"
            tokenizer=tokenizer,
            group_size=128,                # 128 is standard; smaller = more accurate, slower
            damp_percent=0.1,
            desc_act=False,                # True can improve accuracy but is slower
            sym=True,                      # symmetric quantization
            true_sequential=True,          # quantize layer-by-layer (less RAM)
        )

        print(f"{BLUE}[Step 2] Loading + quantizing {RED}{model_path}{RESET} (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=gptq_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        quantized_path = model_path.replace("_fp16", f"_{quantization}")
        print(f"{BLUE}[Step 3] Saving to {GREEN}{quantized_path}{RESET} ...")
        model.save_pretrained(quantized_path, safe_serialization=True)
        tokenizer.save_pretrained(quantized_path)
    else:   
        raise Exception(f"Unhandled quantization method {quantization}")
    print("✅ Quantization complete!")
    
    
def _quantize_mistral_model(model_path, device_map, quantization:Literal["bnb_4bit", "gptq_2bit", "gptq_3bit","gptq_4bit","gptq_8bit"]="bnb_4bit", calibration_set:list[str]=None):
    
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
            device_map=device_map,
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
    elif quantization.startswith("gptq_"):
        _ensure_gptq_backend_available(device_map)
        from transformers import GPTQConfig
        print(f"{BLUE}[Step 1] Loading {RED}(model) {model_path}{RESET} ...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        gptq_config = GPTQConfig(
            bits=int(quantization[5]),                        # 2, 3, 4, or 8
            dataset=calibration_set,       # list[str] or a dataset name like "c4"
            tokenizer=tokenizer,
            group_size=128,                # 128 is standard; smaller = more accurate, slower
            damp_percent=0.1,
            desc_act=False,                # True can improve accuracy but is slower
            sym=True,                      # symmetric quantization
            true_sequential=True,          # quantize layer-by-layer (less RAM)
        )
        
        print(f"{BLUE}[Step 2] Loading + quantizing {RED}{model_path}{RESET} (this may take a while)...")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            offload_folder="./offload_flashml",
            quantization_config=gptq_config,
            trust_remote_code=True,
            )
        
        quantized_path = model_path.replace("_fp16", f"_{quantization}")
        print(f"{BLUE}[Step 3] Saving to {GREEN}{quantized_path}{RESET} ...")
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
        raise Exception(f"Unhandled quantization method {quantization} for mistral models")
    print("✅ Quantization complete!")
