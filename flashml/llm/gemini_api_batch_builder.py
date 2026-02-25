from datetime import datetime
import os

# https://ai.google.dev/gemini-api/docs/batch-api
class GeminiBatchRequest():
    def __init__(self, api_key, messages_batch:list[list[dict]], model_name:str, config):
        """_summary_

        Args:
            api_key (_type_): _description_
            messages_batch (list[list[dict]]): { "content": [{"parts": [{"text": "Hello, world!"}}]}]}
            model_name (str): _description_
            config (_type_): _description_
        """
        from google import genai
        from google.genai import types
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.message_batch = messages_batch
        self.current_file_name = None
        self.current_uploaded_file = None
        
    def step_1_build_batch_file(self, file_name=None, custom_ids:list = None):
        if file_name is None:
            base_name = f"gemini_batch_req_file_{datetime.now().strftime('%d%m')}"
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
            request = {
                "key": f"async-req-{elem_id+1}" if custom_ids is None else str(custom_ids[elem_id]), # it must be str
                "request": mess
            }
            log_json(record=request, path=file_name, add_timestamp=True)
            
        request_id_mode = "w/ custom IDs" if custom_ids is not None else "w/ default request IDs"
        structured_output_suffix = ", w/ structured output" if getattr(self, "output_structure", None) is not None else ""
        print(f"[1] \x1b[38;5;33mGemini\x1b[38;5;32m Batch File (File Name: x1b[38;5;39m{file_name}\x1b[38;5;32m) created locally (num_requests={len(self.messages_batch)}, {request_id_mode}{structured_output_suffix}).\x1b[37m")
        
    def step_2_upload_batch_file(self, file_name="current"):
        from google.genai import types
        if file_name == "current":
            if self.current_file_name == None:
                raise Exception(f"There is no current batch file name. Please select a correct file_name to upload. (received: {file_name})")
            file_name = self.current_file_name
            
        self.current_uploaded_file = self.client.upload(
            file=file_name,
            config=types.UploadFileConfig(display_name=os.path.splitext(file_name)[0], mime_type='jsonl')
        )
        
        print(f"[2] \x1b[38;5;33mGemini\x1b[38;5;32m Batch File (File ID: \033[32m{self.current_uploaded_file.name}\x1b[38;5;32m) uploaded.\033[37m")
        
    def step_3_create_batch_job(self, file_name="current", metadata:dict={}):
        fn = self.current_uploaded_file.name if file_name == "current" else file_name
        self.CURRENT_batch_job = self.client.batches.create(
            model=self.model_name,
            scr= fn, 
            config={
                "display_name" : f"{fn}-job"
            }
        )
        
        print
