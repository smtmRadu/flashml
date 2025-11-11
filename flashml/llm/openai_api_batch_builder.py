from datetime import datetime
from typing import Literal
import os
class OpenAISyncRequest():
    def __init__(self, api_key:str, messages_batch:list[list[dict]], model_name:str,
                 max_completion_tokens:int = None, temperature:float = None,
                 reasoning_effort: Literal["minimal", "low", "medium", "high"] = None,
                 format=None):
        from openai import Client
        self.model_name = model_name
        self.max_tokens = max_completion_tokens
        self.messages_batch = messages_batch
        self.temperature = temperature
        self.output_structure = format
        self.reasoning_effort = reasoning_effort
        self.client = Client(api_key=api_key)
        
    def get_responses(self, file_name=None, custom_ids: list=None):
        if file_name is None:
            base_name = f"openai_sync_req_output_{datetime.now().strftime('%d%m')}"
            version = 1
            file_name = f"{base_name}_v{version}.jsonl"

            # Increment version number until a non-existing filename is found
            while os.path.exists(file_name):
                version += 1
                file_name = f"{base_name}_v{version}.jsonl"
                
        from tqdm import tqdm
        from flashml import log_json
        with open(file_name, "w") as f:
            f.write("")
            
        print(f"\033[32mOpenAI\033[38;2;189;252;201m Output File (File Name: \033[38;2;0;128;128m{file_name}\033[38;2;189;252;201m) created locally (num_requests={len(self.messages_batch)}) {'w/ custom IDs' if custom_ids is not None else 'w/ default request IDs'}{', w/ structured output' if self.output_structure is not None else ''}.\033[37m")
        responses = []
        for idx, mess in tqdm(enumerate(self.messages_batch), total=len(self.messages_batch), desc=f"Getting {self.model_name} responses"):
            if self.output_structure is None:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=mess,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort
                )
            else:
                resp = self.client.chat.completions.parse(
                    model=self.model_name,
                    messages=mess,
                    response_format=self.output_structure,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort
                )

            msg = resp.choices[0].message
            if hasattr(msg, "to_dict"):
                msg = msg.to_dict()

            elm = {
                "custom_id": f"sync-req-{idx+1}" if custom_ids is None else str(custom_ids[idx]),
                "message": msg
            }
            log_json(record=elm, path=file_name, add_timestamp=True)
            responses.append(elm)

        return responses

    

class OpenAIBatchRequest():
    def __init__(self, api_key:str, messages_batch:list[list[dict]], model_name:str,max_tokens:int = None, temperature:float = None, reasoning_effort: Literal["minimal", "low", "medium", "high"] = None,format=None):
        """Initializes the BatchedRequest client. Call the each function in step order. (_step1_(), _step2_(), _step3_() and _step4_())
        
Note that when doing structured output in batched inference, the output is extracted directly as a json as follows:
```python
resps = load_records(BATCH_OUTPUT_FILE_NAME)
jsons = []
for r in resps:
    js = json.loads(r["response"]["body"]['choices'][0]['message']["content"])
    jsons.append(js)
```
        Args:
            client (_type_): _description_
            messages_batch (list[list[dict]]): _description_
            model_name (str): _description_
            max_tokens (int): _description_
            temperature (float): _description_
            output_structure (Type[BaseModel], optional): _description_. Defaults to None.
        """
        from openai import Client
        self.model_name= model_name
        self.max_tokens = max_tokens
        self.messages_batch= messages_batch
        self.temperature= temperature
        self.reasoning_effort = reasoning_effort
        self.client = Client(api_key=api_key)
        self.output_structure = format
        self.current_file_name = None
        
    def step_1_build_batch_file(self, file_name = None, custom_ids: list = None):
        if file_name is None:
            base_name = f"openai_batch_req_file_{datetime.now().strftime('%d%m')}"
            version = 1
            file_name = f"{base_name}_v{version}.jsonl"

            # Increment version number until a non-existing filename is found
            while os.path.exists(file_name):
                version += 1
                file_name = f"{base_name}_v{version}.jsonl"
        
        self.current_file_name = file_name
        
        from flashml import log_json
        
        
        with open(file_name, "w") as f:
            f.write("")
            
        for elem_id, mess in enumerate(self.messages_batch):
            req = {
                "custom_id" : f"async-req-{elem_id+1}" if custom_ids is None else str(custom_ids[elem_id]), #note custom_ids must be string as openai api says
                "method": "POST",
                "url": "/v1/chat/completions",
                "body":{
                    "model": self.model_name,
                    "messages": mess,
                    "reasoning_effort": self.reasoning_effort
                    }
                }
            if self.max_tokens:
                req["body"]["max_completion_tokens"] = self.max_tokens
            if self.temperature:
                req["body"]["temperature"] = self.temperature    
            
            if self.output_structure is not None:
                schema = self.output_structure.model_json_schema()
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
            log_json(record=req, path=file_name, add_timestamp=False)
            
        print(f"[1] \033[32mOpenAI\033[38;2;189;252;201m Batch File (File Name: \033[32m{file_name}\033[38;2;189;252;201m) created locally (num_requests={len(self.messages_batch)}, {"w/ custom IDs" if custom_ids is not None else "w/ default request IDs"}{", w/ structured output" if self.output_structure is not None else ""}).\033[37m")

    def step_2_upload_batch_file(self, file_name = "current"):
        if file_name == "current":
            if self.current_file_name == None:
                raise Exception(f"There is no current batch file name. Please select a correct file_name to upload. (received: {file_name})")
            file_name = self.current_file_name
            

        input_file = self.client.files.create(
                file = open(file_name, "rb"),
                purpose="batch"
        )
        self.CURRENT__file_id = input_file.id
        print(f"[2] \033[32mOpenAI\033[38;2;189;252;201m Batch File (File ID: \033[32m{input_file.id}\033[38;2;189;252;201m) uploaded.\033[37m")
        
    def step_3_create_batch(self, file_id:str="current", metadata:dict = {}):
        from flashml import bell
        """Puts the batch into processing. You can cancel the batch at anytime. 

        Args:
            file_id (str, optional): _description_. Defaults to "current".
            metadata (dict, optional): _description_. Defaults to {}.
            output_structure (Type[BaseModel], optional): _description_. Defaults to None.
        """
        bell()
        # if input(f"This will create the batch (of size {len(self.messages_batch)} elements) and will start to process. Are you sure you want to continue? (y/n)").lower() == "y":

        self.CURRENT__batch = self.client.batches.create(
                input_file_id=self.CURRENT__file_id if file_id=="current" else file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata
            )

        print(f"[3] \033[32mOpenAI\033[38;2;189;252;201m Batch created (Batch ID: \033[32m{self.CURRENT__batch.id}\033[37m, File ID: \033[32m{self.CURRENT__file_id if file_id=="current" else file_id}\033[37m). \033[38;2;189;252;201mSee: https://platform.openai.com/batches.\033[37m")
    
    def step_4_poll_batch(self, batch_id = "current", check_every_s: float = 5):
        """
        This function makes requests to the api and checks the progress of the batch. It retrieves the batch output when is ready. You can also download it from from the web.
        Args:
            check_every_s (1): _description_

        Returns:
            The retrieved batch when completed
        """
        from datetime import datetime
        from flashml import bell
        from time import sleep
        while True:
            batch = self.client.batches.retrieve(self.CURRENT__batch.id if batch_id=="current" else batch_id)
            total = batch.request_counts.total if batch.request_counts.total > 0 else 1
            print("[4] \033[32mOpenAI\033[37m |",
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "|",
                  f" status: \033[38;2;189;252;201m{batch.status}\033[0m (\033[38;2;0;128;128m{(batch.request_counts.completed + batch.request_counts.failed) * 100/total:.2f}%\033[0m)", "|", 
                  f"({batch.request_counts.completed} \033[38;5;10mcompleted\033[0m" ,"|",
                  f"{batch.request_counts.failed} \033[38;5;9mfailed\033[0m" , "|",
                  f"{batch.request_counts.total} \033[38;5;12mtotal\033[0m) ",  "|", 
                  f" Batch ID: \033[38;2;189;252;201m{batch.id}\033[37m")
            if batch.status == "completed":
                bell()
                break
            sleep(check_every_s)
        return self.client.batches.retrieve(self.CURRENT__batch.id if batch_id=="current" else batch_id)
       
    def cancel_batch(self, batch_id):
        """
        Note this can be done also from OpenAI website.
        """
        self.client.cancel(batch_id)
        
    def print_files(self, limit=10):
        for i in self.client.files.list(limit=limit):
            print(i)
            
    def print_batches(self,limit=10):
        for i in self.client.batches.list(limit=limit):
            print(i)