
from typing import Literal

def merge_llm(
    adapter_path: str,
    base_model_path: str = "auto",
    dtype: Literal["fp16", "bf16", "fp32"] = "bf16"
):
    """
    Merges a base model with a LoRA adapter and saves it in the specified precision
    (fp16, bf16, or fp32). No quantization is performed.

    Args:
        adapter_path (str): Path to LoRA adapter
        base_model_path (str): Path to base model (if not specified, inferred from adapter_config.json)
        dtype (str): Precision type ("fp16", "bf16", "fp32")

    Example:
        merge_llm("./my_adapter", "Qwen/Qwen3-7B-Instruct", dtype="bf16")
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from pathlib import Path
    import json, os, shutil

    # --- Dtype handling ---
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_map[dtype]

    # --- Output path ---
    adapter_path_obj = Path(adapter_path)
    suffix = f"_{dtype}"
    save_path = adapter_path_obj.parent / f"{adapter_path_obj.name}_merged{suffix}"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Step 1/2: Loading and merging models ({dtype})...")

    # --- Infer base model if needed ---
    if base_model_path == "auto":
        config_path = adapter_path_obj / "adapter_config.json"
        if not config_path.exists():
            raise ValueError("Base model path not specified and adapter_config.json not found.")
        with open(config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError("'base_model_name_or_path' missing from adapter_config.json")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # --- Load and merge ---
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch_dtype,
        offload_folder="./offload_flashml",
        trust_remote_code=True
    )

    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
        offload_folder="./offload_flashml"
    ).merge_and_unload()

    print("✓ Models merged successfully")

    # --- Save merged model ---
    print(f"💾 Step 2/2: Saving merged model in {dtype.upper()} precision...")
    merged_model.to(dtype=torch_dtype)
    merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    # --- Copy extra files ---
    for fname in ["chat_template.jinja", "README.md", "training_args.bin"]:
        src = adapter_path_obj / fname
        dst = save_path / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"📎 Copied {fname} to merged directory")

    # --- Cleanup ---
    if os.path.exists("./offload_flashml"):
        shutil.rmtree("./offload_flashml")

    print(f"\n🎉 Model merged and saved to {save_path} ({dtype})")
    return save_path


def merge_and_quantize_llm(
    adapter_path: str,
    base_model_path: str = "auto",
    quant_type: Literal["gptq", "awq", "bitsandbytes"] = "bitsandbytes",
    calibration_dataset: list[str] | None = None,
    dtype: Literal["fp16", "bf16", "fp32"] = "bf16"
):
    """
    Merges a base model with a LoRA adapter and quantizes it to 4-bit.
    All quantized models are saved in 4-bit format.

    Args:
        adapter_path (str): Path to LoRA adapter
        base_model_path (str): Base model path or "auto" to infer from adapter_config.json
        quant_type (str): "gptq", "awq", or "bitsandbytes"
        calibration_dataset (list[str] | None): Required for GPTQ/AWQ
        dtype (str): Model load/merge precision ("fp16", "bf16", "fp32")
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from pathlib import Path
    import os, shutil, tempfile, json

    quant_type = quant_type.lower()
    if quant_type not in ["gptq", "awq", "bitsandbytes"]:
        raise ValueError(f"Invalid quant_type: {quant_type}")

    # Handle dtypes
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    if quant_type in ["gptq", "awq"] and not calibration_dataset:
        raise ValueError(f"{quant_type.upper()} requires calibration_dataset")

    suffix_map = {
        "awq": "_AWQ",
        "gptq": "_GPTQ",
        "bitsandbytes": "_bnb"
    }

    adapter_path_obj = Path(adapter_path)
    save_path = adapter_path_obj.parent / f"{adapter_path_obj.name}_merged{suffix_map[quant_type]}"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Step 1/3: Loading and merging models...")

    if base_model_path == "auto":
        config_path = adapter_path_obj / "adapter_config.json"
        if not config_path.exists():
            raise ValueError("adapter_config.json missing, cannot infer base model path.")
        with open(config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError("Cannot infer base model path from adapter_config.json")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch_dtype,
        offload_folder="./offload_flashml",
        trust_remote_code=True
    )

    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
        offload_folder="./offload_flashml"
    ).merge_and_unload()

    print("✓ Models merged successfully")
    print(f"\n🔄 Step 2/3: Quantizing with {quant_type.upper()}...")

    if quant_type == "gptq":
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        examples = [
            tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).input_ids
            for text in calibration_dataset
        ]
        quant_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False, damp_percent=0.01)
        merged_model = merged_model.to("cuda")
        gptq_model = AutoGPTQForCausalLM.from_quantized(
            merged_model, quantize_config=quant_config, device_map="auto"
        )
        gptq_model.quantize(examples)
        merged_model = gptq_model

    elif quant_type == "awq":
        from llmcompressor.transformers import compress
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: false
                        strategy: "channel"
                    targets: ["Linear"]
"""
            compress(
                model=temp_dir,
                dataset=calibration_dataset,
                recipe=recipe,
                output_dir=str(save_path),
                num_calibration_samples=min(128, len(calibration_dataset)),
            )
        merged_model = None

    elif quant_type == "bitsandbytes":
        # BitsAndBytes requires calibration dataset for proper 4-bit serialization
        if not calibration_dataset:
            print("⚠️  Warning: BitsAndBytes without calibration_dataset will save config only.")
            print("    The model will quantize at runtime but won't have compressed weights.")
        
        from transformers import BitsAndBytesConfig
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Save with BitsAndBytes config for runtime quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True
            )
            
            # Load quantized model
            merged_model = AutoModelForCausalLM.from_pretrained(
                temp_dir,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )

    print("✓ Quantization complete")
    print("\n🔄 Step 3/3: Saving quantized model...")

    if merged_model is not None:
        merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    for fname in ["chat_template.jinja", "README.md", "training_args.bin"]:
        src = adapter_path_obj / fname
        dst = save_path / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"📎 Copied {fname} to merged directory")

    if os.path.exists("./offload_flashml"):
        shutil.rmtree("./offload_flashml")

    print(f"\n🎉 Model merged, quantized, and saved to {save_path}")
    return save_path
def get_4bit_quantization_config():
    from transformers import BitsAndBytesConfig
    import torch
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
def get_boxed_answer(text: str) -> str | None:
    '''
    Return the <answer> from the last \\boxed{<answer>} in an LLM response.
    If no \\boxed{} is found, returns None
    '''
    import re
    matches = re.findall(r'\\boxed\{(.+?)\}', text)
    return matches[-1] if matches else None