from .model_inspector_window import inspect_model_window
from .tokenizer_inspector_window import inspect_tokenizer_window
from .model_inspector_notebook import inspect_model_notebook
from .tokenizer_inspector_notebook import inspect_tokenizer_notebook
from typing import Literal



def inspect_model(model, renderer: Literal["tkinter", "vscode"]= "vscode", input_data = None, view_uint8_as_int4: bool = True):
    """Inspects the model either by poping a tkinter window (deprecated or for python use) or in notebook (recommended).

    Args:
        model (_type_): _description_
        renderer (Literal[&quot;tkinter&quot;, &quot;vscode&quot;], optional): _description_. Defaults to "vscode".
        input_data (_type_, optional): _description_. Defaults to None.
        view_uint8_as_int4 (bool, optional): _description_. Defaults to True.
    """
    if renderer == "tkinter":
        inspect_model_window(model)
    else:
        inspect_model_notebook(model, input_data=input_data, renderer=renderer, view_uint8_as_int4=view_uint8_as_int4)
    
def inspect_tokenizer(tokenizer, renderer: Literal["tkinter", "vscode"] = "tkinter"):
    """Inspects a tokenizer either by poping a tkinter window (recommended because has more features) or in a notebook (deprecated, simple view)

    Args:
        tokenizer (_type_): _description_
        renderer (Literal[&quot;tkinter&quot;, &quot;vscode&quot;], optional): _description_. Defaults to "tkinter".
    """
    if renderer == "tkinter":
        inspect_tokenizer_window(tokenizer=tokenizer)
    else:
        inspect_tokenizer_notebook(tokenizer=tokenizer, renderer=renderer)
        
        
__all__ = ["inspect_model", "inspect_tokenizer"]

__all__  == sorted(__all__ ), "Sort this"
