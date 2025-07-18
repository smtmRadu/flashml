from pydantic import BaseModel
from typing import Type


class OpenAISyncRequest():
    def __init__(self, client, messages_batch:list[list[dict]], model_name:str,max_tokens:int, temperature:float, output_structure:Type[BaseModel]=None):
        """Initializes a SyncronizedRequest client.
The output .jsonl contains the following response format:
```
{
    "timestamp" : date
    "custom_id" : str
    "message" : {
        "content" : str (what you actually need, regardless it is structured output or not)
    }
}
```
        Args:
            client (_type_): _description_
            messages_batch (list[list[dict]]): _description_
            model_name (str): _description_
            max_tokens (int): _description_
            temperature (float): _description_
            output_structure (Type[BaseModel], optional): _description_. Defaults to None.
        """
        self.model_name= model_name
        self.max_tokens = max_tokens
        self.messages_batch= messages_batch
        self.temperature= temperature
        self.output_structure= output_structure
        self.client = client
        
    def get_responses(self, file_name = "openai_sync_req_output.jsonl", custom_ids: list= None,):
        from tqdm import tqdm
        from flashml import log_record
        with open(file_name, "w") as f:
            f.write("")
            
        print(f"\033[32mOpenAI\033[38;2;189;252;201m Output File (File Name: \033[38;2;0;128;128m{file_name}\033[38;2;189;252;201m) created locally (num_requests={len(self.messages_batch)}) {"w/ custom IDs" if custom_ids is not None else "w/ default request IDs"}{", w/ structured output" if self.output_structure is not None else ""}.\033[37m")
        responses = []
        for idx, mess in tqdm(enumerate(self.messages_batch),total=len(self.messages_batch), desc=f"Getting {self.model_name} responses"):
            if self.output_structure is None:
                resp = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=mess
                )
            else:
                resp =  self.client.chat.completions.parse(
                    model = self.model_name,
                    messages= mess,
                    response_format = self.output_structure
                )
            elm = {"custom_id" : f"sync-req-{idx+1}" if custom_ids is None else str(custom_ids[idx]), "message": resp.choices[0].message}
            log_record(record=elm, path=file_name, add_timestamp=True)
            responses.append(elm) 

        return responses
    

class OpenAIBatchRequest():
    def __init__(self, client, messages_batch:list[list[dict]], model_name:str,max_tokens:int, temperature:float, output_structure:Type[BaseModel]=None):
        """Initializes the BatchedRequest client.
        
Note that when doing structured output in batched inference, the output is extracted directly as a json as follows:
```python
resps = load_records(BATCH_OUTPUT_FILE_NAME)
jsons = []
for r in resps:
    js = json.loads(r["response"]["body"]['choices'][0]['message']["tool_calls"][0]["function"]["arguments"])
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
        self.model_name= model_name
        self.max_tokens = max_tokens
        self.messages_batch= messages_batch
        self.temperature= temperature

        self.client = client
        self.output_structure = output_structure
        
    def step_1_build_batch_file(self, file_name = "openai_batch_req_file.jsonl", custom_ids: list = None):
        from flashml import log_record
        
        
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
                    "max_tokens":self.max_tokens,
                    "temperature": self.temperature
                    }
                }

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
            log_record(record=req, path=file_name, add_timestamp=False)
            
        print(f"[1] \033[32mOpenAI\033[38;2;189;252;201m Batch File (File Name: \033[38;2;0;128;128m{file_name}\033[38;2;189;252;201m) created locally (num_requests={len(self.messages_batch)}, {"w/ custom IDs" if custom_ids is not None else "w/ default request IDs"}{", w/ structured output" if self.output_structure is not None else ""}).\033[37m")

    def step_2_upload_batch_file(self, file_name = "openai_batch_req_file.jsonl"):
        input_file = self.client.files.create(
                file = open(file_name, "rb"),
                purpose="batch"
        )
        self.CURRENT__file_id = input_file.id
        print(f"[2] \033[32mOpenAI\033[38;2;189;252;201m Batch File (File ID: \033[38;2;0;128;128m{input_file.id}\033[38;2;189;252;201m) uploaded.\033[37m")
        
    def step_3_create_batch(self, file_id:str="current", metadata:dict = {}):
        from flashml import bell
        """Puts the batch into processing. You can cancel the batch at anytime. 

        Args:
            file_id (str, optional): _description_. Defaults to "current".
            metadata (dict, optional): _description_. Defaults to {}.
            output_structure (Type[BaseModel], optional): _description_. Defaults to None.
        """
        bell()
        if input(f"This will create the batch (of size {len(self.messages_batch)} elements) and will start to process. Are you sure you want to continue? (y/n)").lower() == "y":

            self.CURRENT__batch = self.client.batches.create(
                input_file_id=self.CURRENT__file_id if file_id=="current" else file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata
            )
        else:
            print("The batch was not created! Nothing is going to be processed")
            return 
        print(f"[3] \033[32mOpenAI\033[38;2;189;252;201m Batch created (Batch ID: \033[32m{self.CURRENT__batch.id}\033[37m, File ID: \033[32m{self.CURRENT__file_id if file_id=="current" else file_id}\033[37m).\n\033[38;2;189;252;201m\tView progress and retrieve results at at https://platform.openai.com/batches.\033[37m")
    
    def step_4_poll_batch(self, batch_id = "current", check_every_s: int = 1):
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
            print("\033[32mOpenAI\033[37m |",
                  datetime.now(), "|",
                  f" status: {batch.status}", "|", 
                  f"({batch.request_counts.completed} completed" ,"|",
                  f"{batch.request_counts.failed} failed" , "|",
                  f"{batch.request_counts.total} total) ",  "|", 
                  f"Batch ID: \033[38;2;189;252;201m{batch.id}\033[37m")
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