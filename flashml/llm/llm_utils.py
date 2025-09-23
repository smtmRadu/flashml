


def merge_llm_with_adapter(base_model_path:str, adapter_path:str, merge_pathname:str = None, dtype:str="float16"):
    """Don't run this function inside WSL because it crashes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    simplified_dtype = dtype.replace("float", "fp")
    
    if merge_pathname is None:
        merge_pathname = base_model_path + "_merged_" + simplified_dtype
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merge_pathname)
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='cpu', dtype=dtype, offload_folder="./offload_flashml")
    merged_model = PeftModel.from_pretrained(base_model, adapter_path, dtype=dtype, offload_folder="./offload_flashml")
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merge_pathname)
    
    # remove offload folder if exists
    import shutil, os
    if os.path.exists("./offload_flashml"):
        shutil.rmtree("./offload_flashml")
        
    message_start = "🎉 Model successfully merged and saved to "
    message_end = " 🎉"
    colored_path = f"\033[1;36m{merge_pathname}\033[0m"  # Cyan color
    message = message_start + colored_path + message_end

    # Adjust for emoji visual width and ANSI color codes
    visual_adjustment = -3  # Adjust this number if still slightly off
    ansi_codes_length = len("\033[1;36m") + len("\033[0m")  # Length of color codes

    content_line = "┃" + " " * 3 + message + " " * 3 + "┃"
    border_length = len(content_line) - visual_adjustment - ansi_codes_length

    top_line = "┏" + "━" * (border_length - 2) + "┓"
    bottom_line = "┗" + "━" * (border_length - 2) + "┛"

    print(top_line)
    print(content_line)
    print(bottom_line)
    
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