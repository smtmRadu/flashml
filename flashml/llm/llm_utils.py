from typing import Literal    

# NOTE
### When training with unsloth | qlora, the training script uses the unsloth-bnb-4bit as base model for forward pass.
### The adapter result should not be merged with the unsloth-bnb-4bit model (it takes too much time and probably is shit, better with fp16 model), so always pass the fp16 base_model_path for merging.
### There should be no merge_unsloth_llm function.
### 


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