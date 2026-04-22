
import tempfile
import json
import subprocess
import os
import time
from copy import deepcopy
import platform
import sys
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
#  python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-VL-2B-Thinking   --tensor-parallel-size 1 

def vllm_chat_openai_entrypoint(
    messages:list[str] | list[list[str]],
    vllm_config: dict,
    format=None):
    # model:Literal["openai/gpt-oss-120b", "openai/gpt-oss-20b"]="openai/gpt-oss-120b",
    # max_model_len = 131072,
    # max_completion_tokens = 40960,
    # temperature=1.0,
    # top_k=-1,
    # top_p = 1.0,
    # gpu_memory_utilization=0.95,
    # tensor_parallel_size:int=1,
    # reasoning_effort="high",
    # ignore_patterns=["original/**", "metal/**", "consolidated.safetensors"],
    # format=None,
    # other_args:list[str] = ["--max-num-seqs", 256]):
    """
    Check other_args here: 
    https://docs.vllm.ai/en/latest/cli/run-batch.html?utm_source=chatgpt.com#schedulerconfig
    
    Retrieve the output texts as follows:
    
     ```python
    from pydantic import BaseModel
    class OutputFormat(BaseModel):
        output: str
    outputs= openai_vllm_chat(..., format=OutputFormat)
    for outp in outputs:
        response = outp['response']['body']['choices'][0]['message']['content']
        reasoning = outp['response']['body']['choices'][0]['message']['reasoning_content']
        refusal = outp['response']['body']['choices'][0]['message']['refusal']
    ```
    
    Messages may contain None values.

    An autosave JSONL file is written alongside the output file as results
    stream in.  If the batch is interrupted, the autosave contains all
    completed responses.  On success, the autosave is deleted automatically.
    """
    if platform.system() != 'Linux':
        raise OSError(f"vLLM is only supported on Linux. Current system: {platform.system()}")
    
    
    vllm_config = deepcopy(vllm_config)
    #if not os.path.exists(vllm_config["model"]):
    #    raise Exception("The model path does not exist. Please provide the correct path to the model.")
    if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
        messages = [messages]
        
    if len(messages) == 0:
        raise Exception("Input messages list has length 0.")
    
    non_none_messages = [i for i in messages if i is not None]
    vllm_req = []
    for idx, conv in enumerate(non_none_messages):
        req = {
                "custom_id": f"request-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body" :
                    {
                        "model": vllm_config["model"],
                        "messages": conv,
                        "max_completion_tokens": vllm_config["max_completion_tokens"],
                        "temperature": vllm_config["temperature"],
                        "top_p": vllm_config["top_p"] if "top_p" in vllm_config.keys() else 1,
                        "top_k": vllm_config["top_k"] if "top_k" in vllm_config.keys() else -1,
                        "reasoning_effort": vllm_config["reasoning_effort"] if "reasoning_effort" in vllm_config.keys() else None,
                        
                    }
            }
        
        if format is not None:
                schema = format.model_json_schema()
                req["body"]["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": schema["title"],
                        "description": schema.get("description", "Structured output format"),
                        "parameters": schema
                    }
                }]
                req["body"]["tool_choice"] = {
                    "type": "function",
                    "function": {"name": schema["title"]}
                }
            
        vllm_req.append(req)    
         
    input_fd, input_file_path = tempfile.mkstemp(suffix='.jsonl', prefix='batch_input_')
    output_fd, output_file_path = tempfile.mkstemp(suffix='.jsonl', prefix='batch_output_')
      
    os.close(input_fd)
    os.close(output_fd)     
    with open(input_file_path, 'w') as f:
        for rq in vllm_req:
                f.write(json.dumps(rq) + '\n')

    vllm_config.pop("max_completion_tokens", None)
    vllm_config.pop("temperature", None)
    vllm_config.pop("top_p", None)
    vllm_config.pop("top_k", None)
    vllm_config.pop("reasoning_effort", None)
    try:
        # Run vLLM batch processing
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.run_batch",
            "--input-file", input_file_path,
            "--output-file", output_file_path,
        ]
        
        for k, v in vllm_config.items():
            cmd.append(f"--{k}")

            if v is None or v == "":
                continue

            if isinstance(v, (list, tuple)):
                for item in v:
                    cmd.append(str(item))
            else:
                cmd.append(str(v))
                
                
        autosave_path = output_file_path + ".autosave"
        print(f"Instantiating vLLM: \033[92m{' '.join(cmd)}\033[0m")
        print(f"Autosave: {autosave_path}")

        # Stream results — tail the output file while vLLM runs,
        # mirror every completed response to the autosave file.
        process = subprocess.Popen(cmd, text=True)
        seen = 0
        with open(output_file_path, 'r') as tail, \
             open(autosave_path, 'w') as autosave:
            while process.poll() is None:
                line = tail.readline()
                if line.strip():
                    autosave.write(line)
                    autosave.flush()
                    seen += 1
                    sys.stdout.write(
                        f"\r  \033[96mCompleted: {seen}/{len(vllm_req)}\033[0m"
                    )
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
            # Drain any remaining lines after process exits
            for line in tail:
                if line.strip():
                    autosave.write(line)
                    autosave.flush()
                    seen += 1
        sys.stdout.write(
            f"\r  \033[96mCompleted: {seen}/{len(vllm_req)}\033[0m\n"
        )
        sys.stdout.flush()

        if process.returncode != 0:
            print(f"\033[93mProcess failed. Autosave with {seen}/{len(vllm_req)} "
                  f"results at: {autosave_path}\033[0m")
            raise subprocess.CalledProcessError(process.returncode, cmd)

        print(f"\033[92m============== 100% Completed | {len(vllm_req)}/{len(vllm_req)} ==============\033[0m")

        # Read all results for return value
        without_none_responses = []
        with open(output_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    response_data = json.loads(line)
                    without_none_responses.append(response_data)

        with_none_responses = []
        index_in_outp = 0
        for m in messages:
            if m is None:
                with_none_responses.append(None)
            else:
                with_none_responses.append(without_none_responses[index_in_outp])
                index_in_outp += 1

        # Success — autosave no longer needed
        try:
            os.unlink(autosave_path)
        except OSError:
            pass

        return with_none_responses

    except subprocess.CalledProcessError as e:
        print(f"Error running batch processing: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        # Cleanup temporary files (autosave survives on failure)
        try:
            os.unlink(input_file_path)
        except OSError:
            pass
        try:
            os.unlink(output_file_path)
        except OSError:
            pass
        