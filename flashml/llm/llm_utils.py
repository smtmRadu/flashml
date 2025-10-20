
from typing import Literal

def merge_and_quantize_llm(
    adapter_path: str,
    base_model_path: str = "auto",
    quant_type: Literal["gptq", "awq", "bitsandbytes"] = "bitsandbytes",
    calibration_dataset: list[str] | None = None,
    dtype: str = "float16"
):
    """
    Merges a base model with a LoRA adapter and quantizes it in one step.
    Only saves the final quantized model (no intermediate fp16 save).
    
    Don't run this function inside WSL because it crashes.
    
    Args:
        base_model_path (str): Path to base model (if not specified, it is inferred from adapter_config.json inside the adapter_path folder)
        adapter_path (str): Path to LoRA adapter
        calibration_dataset (list[str] | None): Calibration texts for GPTQ/AWQ
        quant_type (str): "gptq", "awq", or "bitsandbytes"
        dtype (str): Data type for merging (default: "float16")
        
    Example:
        # BitsAndBytes (no calibration needed)
        merge_and_quantize_llm(
            "Qwen/Qwen3-7B-Instruct",
            "./my_adapter",
            quant_type="bitsandbytes"
        )
        
        # AWQ with calibration
        calib_data = ["Sample text 1", "Sample text 2", ...]
        merge_and_quantize_llm(
            "Qwen/Qwen3-7B-Instruct",
            "./my_adapter",
            calib_data,
            quant_type="awq"
        )
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from pathlib import Path
    import shutil
    import os
    
    quant_type = quant_type.lower()
    
    if quant_type not in ["gptq", "awq", "bitsandbytes"]:
        raise ValueError(f"Invalid quant_type: {quant_type}. Must be 'gptq', 'awq', or 'bitsandbytes'")
    
    # Validate calibration data for methods that need it
    if quant_type in ["gptq", "awq"] and not calibration_dataset:
        raise ValueError(f"{quant_type.upper()} requires calibration_dataset. Use bitsandbytes for calibration-free quantization.")
    
    # Set output path
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
        # Try to infer base model from adapter's config
        import json
        # Try to infer base model from adapter's config
        config_path = adapter_path_obj / "adapter_config.json"
        if not config_path.exists():
            raise ValueError(
                f"\033[91m[ERROR]\033[0m Base model path not specified "
                f"and adapter_config.json not found in {adapter_path}"
            )
        
        with open(config_path, 'r') as f:
            adapter_config = json.load(f)
        
        if "base_model_name_or_path" in adapter_config:
            base_model_path = adapter_config["base_model_name_or_path"]
            print(
                f"\033[92m[INFO]\033[0m Inferred base model path from adapter: "
                f"\033[96m{base_model_path}\033[0m"
            )
        else:
            raise ValueError(
                "\033[91m[ERROR]\033[0m 'base_model_name_or_path' not found in adapter_config.json. "
                "Please specify base_model_path explicitly."
            )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model and merge with adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map='cpu',
        torch_dtype=dtype,
        offload_folder="./offload_flashml",
        trust_remote_code=True
    )
    
    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=dtype,
        offload_folder="./offload_flashml"
    )
    
    merged_model = merged_model.merge_and_unload()
    
    print(f"✓ Models merged successfully")
    print(f"\n🔄 Step 2/3: Quantizing with {quant_type.upper()}...")
    
    # Quantize the merged model directly
    if quant_type == "gptq":
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            # Prepare calibration examples
            examples = [
                tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=min(2048, tokenizer.model_max_length),
                ).input_ids
                for text in calibration_dataset
            ]
            
            # Configure GPTQ
            quant_config = BaseQuantizeConfig(
                bits=4,
                group_size=128,
                desc_act=False,
                damp_percent=0.01
            )
            
            # Move merged model to appropriate device and quantize
            merged_model = merged_model.to('cuda')
            
            # Wrap with GPTQ quantizer
            gptq_model = AutoGPTQForCausalLM.from_quantized(
                merged_model,
                quantize_config=quant_config,
                device_map="auto"
            )
            
            gptq_model.quantize(examples)
            merged_model = gptq_model
            
        except TypeError as e:
            if "isn't supported yet" in str(e):
                print(f"\n⚠️  GPTQ Error: {e}")
                print("\nQwen3 is not yet supported by auto-gptq.")
                print("RECOMMENDATION: Use BitsAndBytes instead (no calibration needed)")
                raise RuntimeError("GPTQ doesn't support Qwen3 yet. Use 'bitsandbytes' instead.")
            raise
            
    elif quant_type == "awq":
        try:
            from llmcompressor.transformers import compress
            import tempfile
            
            # Save merged model to temp directory for llm-compressor
            with tempfile.TemporaryDirectory() as temp_dir:
                print("Saving merged model to temporary directory...")
                merged_model.save_pretrained(temp_dir)
                tokenizer.save_pretrained(temp_dir)
                
                # AWQ quantization recipe
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
                
                # Quantize with llm-compressor
                compress(
                    model=temp_dir,
                    dataset=calibration_dataset,
                    recipe=recipe,
                    output_dir=str(save_path),
                    max_seq_length=2048,
                    num_calibration_samples=min(128, len(calibration_dataset)),
                )
                
            # Model already saved by compress, skip the save step later
            merged_model = None
            
        except (ImportError, AttributeError) as e:
            print(f"llm-compressor failed ({e}). AutoAWQ doesn't support Qwen3 yet.")
            print("\nRECOMMENDATION: Use GPTQ or BitsAndBytes instead for Qwen3.")
            print("Qwen3 AWQ support is limited. Consider:")
            print("  1. Use pre-quantized models: Qwen/Qwen3-4B-AWQ or Qwen/Qwen3-32B-AWQ")
            print("  2. Use GPTQ quantization (better Qwen3 support)")
            print("  3. Use BitsAndBytes (easiest, no calibration needed)")
            raise RuntimeError(
                "AWQ quantization not available for Qwen3. "
                "Use 'gptq' or 'bitsandbytes' instead, or load pre-quantized Qwen3-AWQ models."
            )
            
    elif quant_type == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        import torch
        
        # For BitsAndBytes, we need to reload with quantization config
        # Save merged model temporarily
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Saving merged model to temporary directory...")
            merged_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Reload with BitsAndBytes quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            merged_model = AutoModelForCausalLM.from_pretrained(
                temp_dir,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
    
    print(f"✓ Quantization complete")
    print(f"\n🔄 Step 3/3: Saving quantized model...")
    
    # Save the quantized model (if not already saved by AWQ)
    if merged_model is not None:
        merged_model.save_pretrained(save_path)
    
    tokenizer.save_pretrained(save_path)
    
    # Copy auxiliary files (if present)
    extra_files = ["chat_template.jinja", "README.md", "training_args.bin"]
    for fname in extra_files:
        src = adapter_path_obj / fname
        dst = save_path / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"📎 Copied {fname} to merged directory")
    # Cleanup
    if os.path.exists("./offload_flashml"):
        shutil.rmtree("./offload_flashml")
    
    # Success message
    message_start = "🎉 Model merged and quantized, saved to "
    message_end = " 🎉"
    colored_path = f"\033[1;36m{save_path}\033[0m"
    message = message_start + colored_path + message_end
    
    visual_adjustment = -3
    ansi_codes_length = len("\033[1;36m") + len("\033[0m")
    
    content_line = "┃" + " " * 3 + message + " " * 3 + "┃"
    border_length = len(content_line) - visual_adjustment - ansi_codes_length
    
    top_line = "┏" + "━" * (border_length - 2) + "┓"
    bottom_line = "┗" + "━" * (border_length - 2) + "┛"
    
    print()
    print(top_line)
    print(content_line)
    print(bottom_line)
    
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