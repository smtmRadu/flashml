from __future__ import annotations

from typing import List, Literal, Optional, Dict, Iterator, overload
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.cache_utils import DynamicCache
from transformers import TextIteratorStreamer
import threading

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

class ChatbotClient:
    """
    Ollama-style client with native HF KV-caching and true parallel batching.
    Now supports per-conversation cache persistence and streaming!
    """

    def __init__(
        self,
        model_name: str,
        *,
        quantization: Literal["default", "8bit", "4bit"] = "4bit",
        device_map: str | dict = "cuda",
    ):
        self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=_bnb_config(quantization),
            trust_remote_code=True,
        )
        self._gen_defaults = dict(
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        # Stores cache per conversation_id; use 0 for single chat
        self._conversation_caches: Dict[int, tuple] = {}

    @overload
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        stream: Literal[True],
        conversation_id: int = 0,
    ) -> Iterator[Dict[str, str]]: ...
    @overload
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        stream: Literal[False] = False,
        conversation_id: int = 0,
    ) -> Dict[str, str]: ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        stream: bool = False,
        conversation_id: int = 0,
    ):
        if stream:
            return self._chat_stream(
                messages=messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                conversation_id=conversation_id,
            )
        return self._chat_nostream(
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            conversation_id=conversation_id,
        )

    def _build_gen_kwargs(
        self,
        *,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        max_new_tokens: int | None,
    ) -> dict:
        gen_kwargs = self._gen_defaults.copy()
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        return gen_kwargs

    def _prepare(
        self,
        messages: List[Dict[str, str]],
        conversation_id: int = 0,
    ):
        """
        Prepare prompt_ids, attention_mask, cache, and cache_pos for the current conversation.
        Loads cache if exists for conversation_id, else creates new DynamicCache.
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        prompt_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        # If continuing a conversation, use saved cache; else new DynamicCache
        cache = self._conversation_caches.get(conversation_id, None)
        if cache is None:
            cache_obj = DynamicCache()
        else:
            _, cache_obj = cache
        cache_pos = torch.arange(prompt_ids.shape[1], dtype=torch.int64, device=self.model.device)
        return prompt_ids, attention_mask, cache_obj, cache_pos

    def _chat_nostream(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        max_new_tokens: int | None,
        conversation_id: int = 0,
    ) -> Dict[str, str]:
        prompt_ids, attention_mask, cache_obj, cache_pos = self._prepare(messages, conversation_id)
        gen_kwargs = self._build_gen_kwargs(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        outputs = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            cache_position=cache_pos,
            past_key_values=cache_obj,
            use_cache=True,
            **gen_kwargs,
        )
        incr_len = prompt_ids.shape[1]
        answer_ids = outputs.sequences[0, incr_len:]
        reply_text = (
            self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        )
        # Update conversation cache
        self._conversation_caches[conversation_id] = (prompt_ids, cache_obj)
        return {"role": "assistant", "content": reply_text}

    def _chat_stream(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        max_new_tokens: int | None,
        conversation_id: int = 0,
    ) -> Iterator[Dict[str, str]]:
        prompt_ids, attention_mask, cache_obj, cache_pos = self._prepare(messages, conversation_id)
        gen_kwargs = self._build_gen_kwargs(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def _generate():
            self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                cache_position=cache_pos,
                past_key_values=cache_obj,
                use_cache=True,
                streamer=streamer,
                **gen_kwargs,
            )
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        for text_chunk in streamer:
            yield {"role": "assistant", "content": text_chunk}

        thread.join()
        # Update cache after streaming for this conversation
        self._conversation_caches[conversation_id] = (prompt_ids, cache_obj)


    def chat_parallel(
        self,
        batches: List[List[dict]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
    ) -> List[str]:
        """
        Generate replies for many independent conversations **in one call**.

        NOTE
        ----
        • KV-caching across calls *per conversation* is not reused here
          because stacking heterogeneous caches is non-trivial; generation
          itself (the expensive part) is parallelised.
        • If you need dialogue-level caching, keep separate `ChatbotClient`
          instances per user.
        """
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
            for msgs in batches
        ]
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.model.device)
        attention_mask = tok.attention_mask
        prompt_lens = attention_mask.sum(dim=1).tolist()
        gen_kwargs = self._gen_defaults.copy()
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        outputs = self.model.generate(
            input_ids=tok.input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        replies: List[str] = []
        for i, prompt_len in enumerate(prompt_lens):
            ans_ids = outputs.sequences[i, prompt_len:]
            replies.append(
                self.tokenizer.decode(
                    ans_ids, skip_special_tokens=True
                ).strip()
            )
        return replies

if __name__ == "__main__":
    bot = ChatbotClient("ibm-granite/granite-3.3-2b-instruct", quantization="4bit")

    hist = [{"role": "user", "content": "Who are you?"}]
    for chunk in bot.chat(hist, stream=True):
        print(chunk["content"], end="", flush=True)

    # A follow-up preserves cache:
    hist2 = hist + [{"role": "assistant", "content": "I'm a language model."}, {"role": "user", "content": "What can you do?"}]
    print("\n---\n")
    for chunk in bot.chat(hist2, conversation_id=cid, stream=True):
        print(chunk["content"], end="", flush=True)
