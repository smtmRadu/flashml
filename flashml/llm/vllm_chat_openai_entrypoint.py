
import tempfile
import json
import subprocess
import os
from typing import Literal

QWEN3_06B_DEFAULT_ARGS = {
    "model": "Qwen/Qwen3-0.6B",
    "max_model_len": 8192,
    "max_completion_tokens": 4096,
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "other_args": ["--quantization", "bitsandbytes"],
}
GPT_OSS_120B_DEFAULT_ARGS = {
    "model": "openai/gpt-oss-120b",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 1,
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "high",
    "ignore_patterns": ["original/**", "metal/**"]
}

MAGISTRAL_SMALL_2506_DEFAULT_ARGS = {
    "model": "mistralai/Magistral-Small-2506",
    "max_model_len": 40_960,
    "max_completion_tokens": 30_720,
    "gpu_memory_utilization": 0.95,
    "tensor_parallel_size": 1,
    "temperature": 0.7,
    "top_p": 0.95,
    "reasoning_effort": "high",
    "ignore_patterns": ["original/**", "metal/**"],
    "other_args": ["--tokenizer-mode", "mistral","--config_format", "mistral", "--load_format", "mistral"]
}

### MAGISTRAL ARGS
# model = "mistralai/Magistral-Small-2506"
# temperature = 0.7
# top_p = 0.95
# max_model_len = 40_960
# max_completion_tokens = 20480
# other_args = ["--tokenizer-mode", "mistral","--config_format", "mistral", "--load_format", "mistral"]
def vllm_chat_openai_entrypoint(
    messages:list[str] | list[list[str]],
    model:Literal["openai/gpt-oss-120b", "openai/gpt-oss-20b"]="openai/gpt-oss-120b",
    max_model_len = 131072,
    max_completion_tokens = 40960,
    temperature=1.0,
    top_k=-1,
    top_p = 1.0,
    gpu_memory_utilization=0.95,
    tensor_parallel_size:int=1,
    reasoning_effort="high",
    ignore_patterns=["original/**", "metal/**", "consolidated.safetensors"],
    format=None,
    other_args:list[str] = ["--max-num-seqs", 256]):
    """
    Check other_args here: 
    https://docs.vllm.ai/en/latest/cli/run-batch.html?utm_source=chatgpt.com#schedulerconfig
    
    If model is not downloaded, run this script first:
    ```python
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            ignore_patterns=["original/**", "metal/**", "consolidated.safetensors"],
            local_dir_use_symlinks=False,  # optional: avoids symlinks
        )
    ```
    Retrieve the output texts as follows:
     ```python
    outputs= openai_vllm_chat(...)
    for outp in outputs:
        response = outp['response']['body']['choices'][0]['message']['content']
        reasoning = outp['response']['body']['choices'][0]['message']['reasoning_content']
    ```
    """
    
    if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
        messages = [messages]
        
    requests = []
    for idx, conv in enumerate(messages):
        req = {
                "custom_id": f"request-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body" :
                    {
                        "model": model,
                        "messages": conv,
                        "max_completion_tokens": max_completion_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "reasoning_effort": reasoning_effort
                        
                    }
            }
        
        if format is not None:
                schema = format.model_json_schema()
                req["body"]["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": schema["title"],
                        "parameters": schema
                    }
                }]
                req["body"]["tool_choice"] = {
                    "type": "function",
                    "function": {"name": schema["title"]}
                }
                req["body"]["response_format"] = {"type": "json_object"}
            
        requests.append(req)    
         
    input_fd, input_file_path = tempfile.mkstemp(suffix='.jsonl', prefix='batch_input_')
    output_fd, output_file_path = tempfile.mkstemp(suffix='.jsonl', prefix='batch_output_')
      
    os.close(input_fd)
    os.close(output_fd)     
    with open(input_file_path, 'w') as f:
        for rq in requests:
                f.write(json.dumps(rq) + '\n')

    try:
        # Run vLLM batch processing
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.run_batch",
            "--input-file", input_file_path,
            "--output-file", output_file_path,
            "--model", model,
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--tensor-parallel-size", str(tensor_parallel_size),
            *[str(x) for x in other_args],
            "--ignore_patterns", *ignore_patterns
        ]

        _ = subprocess.run(cmd, text=True, check=True)
        print(f"\033[92m============== 100% Completed | {len(requests)}/{len(requests)} ==============\033[0m")
        
        # Read results
        responses = []
        with open(output_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    response_data = json.loads(line)
                    responses.append(response_data)
        
        return responses
        
    except subprocess.CalledProcessError as e:
        print(f"Error running batch processing: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        # Cleanup temporary files
        try:
            os.unlink(input_file_path)
            os.unlink(output_file_path)
        except:
            pass