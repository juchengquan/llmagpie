from typing import Any, Dict, List, Mapping, Optional

import requests
# from langchain_core.callbacks import CallbackManagerForLLMRun
# from langchain_core.language_models.llms import LLM
# from langchain_core.pydantic_v1 import Extra, root_validator
# from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel
# typing 
from typing import Optional, List

class JibberLLMNonStream(BaseModel): # TODO: import langchain # (LLM):
    endpoint_url: str = ""
    
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        # run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # _model_kwargs = self.model_kwargs or {} # TODO
        _model_kwargs = {}

        # payload samples
        params = {**_model_kwargs, **kwargs}
        parameter_payload = {"inputs": prompt, "parameters": params}
        

        # HTTP headers for authorization
        headers = {
            # "Authorization": f"Bearer {self.huggingfacehub_api_token}",
            "Content-Type": "application/json",
        }
        
        session = requests.Session()
        # session.verify = False # TODO
        
        try:
            response = session.post(
                self.endpoint_url, headers=headers, json=parameter_payload
            )
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise ValueError(f"Error raised by inference endpoint: {e}")
        
        generated_text = response.json()
        
        return generated_text
            
        
        

jb = JibberLLMNonStream(
    endpoint_url = "http://127.0.0.1:8080/generate"
)
print(jb._call(
    ""
))