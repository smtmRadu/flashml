
import tempfile
import json
import subprocess
import os

def openai_vllm_chat(
    messages:list[str] | list[list[str]],
    temperature=1.0,
    top_p = 0.95,
    gpu_memory_utilization=0.8,
    max_completion_tokens = 4096,
    reasoning_effort="high",
    format=None,
    openai_api_key = "EMPTY",
    openai_api_base = "http://localhost:8000/v1"):
    """
    Initialize vllm first as follows:
    vllm serve openai/gpt-oss-120b
    
    
    Retrieve the output texts as follows:
    outputs= openai_vllm_chat(...)
    for outp in outputs:
        outp['response']['body']['choices'][0]['message']['content']
    """
    
    BATCH_BUILDING_FILE = "openai_batchbuild_flashml.jsonl"
    RESULTS_FILE = "openai_batchresponse_flashml.jsonl"
    from openai import OpenAI
    client =  OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )
    
    models = client.models.list()
    model = models.data[0].id
    
    # 1. Build json request batch
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
            "--gpu-memory-utilization", str(gpu_memory_utilization)
        ]
        
        result = subprocess.run(cmd, text=True, check=True)
        print("Batch processing completed successfully")
        
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