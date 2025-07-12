from __future__ import annotations

from typing import List, Literal, Optional, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.cache_utils import DynamicCache
from transformers import TextIteratorStreamer
import threading


class ChatbotClient:
    def __init__(
        self,
        model_name: str,
        *,
        quantization: Literal["default", "8bit", "4bit"] = "4bit",
        device_map: str | dict = "auto",
    ):
        # self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=ChatbotClient._bnb_config(quantization),
            trust_remote_code=True,
        )
        self.generation_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        
        self.chat_kv_cache = DynamicCache()
        
        
    def _update_gen_kwargs(
        self,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        max_new_tokens: int | None,
    ) -> dict:
        gen_kwargs = self.generation_kwargs.copy()
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        self.generation_kwargs = gen_kwargs
    
    @staticmethod
    def _bnb_config(
        q: Literal["default", "8bit", "4bit"]
    ) -> Optional[BitsAndBytesConfig]:
        q = q.lower()
        if q == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if q == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_skip_modules=None,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        return None
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
    ):
        self._update_gen_kwargs(temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens)
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def _generate():
            self.model.generate(
                **inputs, 
                past_key_values=self.chat_kv_cache,
                use_cache=False,
                streamer=streamer,
                **self.generation_kwargs,
            )
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        for text_chunk in streamer:
            yield {"role": "assistant", "content": text_chunk}

        thread.join()







if __name__ == "__main__":
    bot = ChatbotClient("ibm-granite/granite-3.3-2b-instruct", quantization="4bit")

    hist = [{"role": "user", "content": "Who are you?"}]
    for chunk in bot.chat(hist):
        print(chunk["content"], end="", flush=True)

