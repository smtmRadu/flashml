from typing import Literal    

# NOTE
### When training with unsloth | qlora, the training script uses the unsloth-bnb-4bit as base model for forward pass.
### The adapter result should not be merged with the unsloth-bnb-4bit model (it takes too much time and probably is shit, better with fp16 model), so always pass the fp16 base_model_path for merging.
### There should be no merge_unsloth_llm function.
### 

def merge_llm(
    adapter_path: str,
    base_model_path: str = "auto",
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
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

    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter path {adapter_path} does not exist.")
    
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

    print(f"🔄 Step 1/2: Loading and merging model ({dtype})...")

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
    else:
        print(f"Using {base_model_path} as the base model.")
        
    if "unsloth-bnb" in base_model_path:
        raise ValueError(f"Unsloth-bnb base model path inferred ({base_model_path}). It is not allowed to merge with Unsloth-bnb models. Please consider the non-unsloth bnb-4bit or fp16 for merging.")
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

def quantize_llm(
    base_model_path: str,
    quant_type: Literal["gptq", "awq", "bnb", "mxfp4"] = "bnb",
    dtype: Literal["fp16", "bf16", "fp32"] = "bf16",
    calibration_dataset: list[str] | None = None
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from pathlib import Path
    import os, shutil, tempfile, json
    
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Validate quantization type
    quant_type = quant_type.lower()
    if quant_type not in ["gptq", "awq", "bnb", "mxfp4"]:
        raise ValueError(f"Invalid quant_type: {quant_type}")

    if quant_type in ["gptq", "awq", "mxfp4"] and not calibration_dataset:
        raise ValueError(f"{quant_type.upper()} requires a calibration_dataset")

    print(f"🔄 Loading base model {base_model_path}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch_dtype,
        offload_folder="./offload_flashml",
        trust_remote_code=True,
    )

    print("✓ Model loaded successfully")

    # Get the directory of the base model path
    save_path = Path(base_model_path).parent
    print(f"Quantized model will be saved to: {save_path}")

    # Perform quantization based on the selected quantization type
    print(f"\n🔄 Step 1/3: Quantizing with {quant_type.upper()}...")

    if quant_type == "gptq":
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        examples = [
            tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).input_ids
            for text in calibration_dataset
        ]
        quant_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False, damp_percent=0.01)
        base_model = base_model.to("cuda")
        gptq_model = AutoGPTQForCausalLM.from_quantized(
            base_model, quantize_config=quant_config, device_map="auto"
        )
        gptq_model.quantize(examples)
        base_model = gptq_model

    elif quant_type == "awq":
        from llmcompressor.transformers import compress
        with tempfile.TemporaryDirectory() as temp_dir:
            base_model.save_pretrained(temp_dir)
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
        base_model = None

    elif quant_type == "mxfp4":
        try:
            from llmcompressor.transformers import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
        except ImportError:
            raise ImportError(
                "MX-FP4 quantization requires llm-compressor. Install with: "
                "pip install llmcompressor[transformers]"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            base_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)

            # MX-FP4 recipe for block-wise microscaling quantization
            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP4",
                ignore=["lm_head"],
                config_groups={
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "strategy": "group",
                            "group_size": 128,
                            "block_structure": "1x128"  # MX block structure
                        }
                    }
                }
            )

            oneshot(
                model=temp_dir,
                dataset=calibration_dataset,
                recipe=recipe,
                output_dir=str(save_path),
                num_calibration_samples=min(512, len(calibration_dataset)),
                max_seq_length=2048,
                pad_to_max_length=False
            )
        base_model = None

    elif quant_type == "bnb":
        from transformers import BitsAndBytesConfig
        with tempfile.TemporaryDirectory() as temp_dir:
            base_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)

            # Load quantized model
            base_model = AutoModelForCausalLM.from_pretrained(
                temp_dir,
                quantization_config=get_bnb_4bit_quantization_config(),
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )

    print("✓ Quantization complete")

    # Save the quantized model and tokenizer
    print(f"\n🔄 Step 2/3: Saving quantized model...")

    base_model.save_pretrained(save_path, safe_serialization=True)
    #tokenizer.save_pretrained(save_path)

    print(f"🎉 Model quantized and saved to {save_path}")
    return str(save_path)

def merge_and_quantize_llm(
    adapter_path: str,
    base_model_path: str = "auto",
    quant_type: Literal["gptq", "awq", "bnb", "mxfp4"] = "bnb",
    calibration_dataset: list[str] | None = None,
    dtype: Literal["fp16", "bf16", "fp32"] = "bf16"
):
    """
    Merges a base model with a LoRA adapter and quantizes it to 4-bit.
    All quantized models are saved in 4-bit format.

    Args:
        adapter_path (str): Path to LoRA adapter
        base_model_path (str): Base model path or "auto" to infer from adapter_config.json
        quant_type (str): "gptq", "awq", "bnb", or "mxfp4"
        calibration_dataset (list[str] | None): Required for GPTQ/AWQ/MXFP4
        dtype (str): Model load/merge precision ("fp16", "bf16", "fp32")
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from pathlib import Path
    import os, shutil, tempfile, json

    
    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter path {adapter_path} does not exist.")
    
    
    quant_type = quant_type.lower()
    if quant_type not in ["gptq", "awq", "bnb", "mxfp4"]:
        raise ValueError(f"Invalid quant_type: {quant_type}")

    # Handle dtypes
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    if quant_type in ["gptq", "awq", "mxfp4"] and not calibration_dataset:
        raise ValueError(f"{quant_type.upper()} requires calibration_dataset")

    suffix_map = {
        "awq": "_AWQ",
        "gptq": "_GPTQ",
        "bnb": "_bnb",
        "mxfp4": "_MXFP4"
    }

    adapter_path_obj = Path(adapter_path)
    save_path = adapter_path_obj.parent / f"{adapter_path_obj.name}_merged{suffix_map[quant_type]}"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Step 1/3: Loading and merging model...")

    if base_model_path == "auto":
        config_path = adapter_path_obj / "adapter_config.json"
        if not config_path.exists():
            raise ValueError("adapter_config.json missing, cannot infer base model path.")
        with open(config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError("Cannot infer base model path from adapter_config.json")
        
    if "4bit" in base_model_path:
        raise ValueError(f"4bit model path inferred ({base_model_path}). Please pass the original HuggingFace 16bit model path so it can be quantized.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch_dtype,
        offload_folder="./offload_flashml",
        trust_remote_code=True,
        quantization_config=None,  # ← Add this
        load_in_4bit=False,        # ← Add this
        load_in_8bit=False         # ← Add this
    )

    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
        offload_folder="./offload_flashml"
    ).merge_and_unload()

    print("✓ Model merged successfully")
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

    elif quant_type == "mxfp4":
        try:
            from llmcompressor.transformers import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
        except ImportError:
            raise ImportError(
                "MX-FP4 quantization requires llm-compressor. Install with: "
                "pip install llmcompressor[transformers]"
            )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # MX-FP4 recipe for block-wise microscaling quantization
            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP4",
                ignore=["lm_head"],
                config_groups={
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "strategy": "group",
                            "group_size": 128,
                            "block_structure": "1x128"  # MX block structure
                        }
                    }
                }
            )
            
            # Perform one-shot quantization with calibration data
            oneshot(
                model=temp_dir,
                dataset=calibration_dataset,
                recipe=recipe,
                output_dir=str(save_path),
                num_calibration_samples=min(512, len(calibration_dataset)),
                max_seq_length=2048,
                pad_to_max_length=False
            )
        merged_model = None

    elif quant_type == "bnb":
        # BitsAndBytes requires calibration dataset for proper 4-bit serialization
        from transformers import BitsAndBytesConfig
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            

            # Load quantized model
            merged_model = AutoModelForCausalLM.from_pretrained(
                temp_dir,
                quantization_config=get_bnb_4bit_quantization_config(),
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



def get_bnb_4bit_quantization_config():
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

def image_to_base64(image):
    """
    How to use it. Load images in PIL format then add this to the content.
    {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {  "url": f"data:image/png;base64,{img_base64}"    },
                    "resized_height": 280,
                    "resized_width": 420
                },
                {"type": "text", "text": 'Describe the image' },
            ],
        }
    """
    from PIL import Image
    import base64
    from io import BytesIO

    """Convert image to base64 string."""
    if isinstance(image, str):
        with Image.open(image) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError("Provided image is neither a valid path nor a PIL.Image object.")
    return img_str