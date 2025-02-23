# allows to load python file (llama parse is not able to handle code)
# read contents of code file and give to llm

from llama_index.core.tools import FunctionTool
# wrap any python function as a tool that we can pass to llm

import os

def code_reader_fn(file_name):
    path = os.path.join("data", file_name)
    try:
        with open(path, "r") as f:
            content = f.read()
            return {"file_content": content}
    except Exception as e:
        return {"error": str(e)}
    
code_reader = FunctionTool.from_defaults(
    fn = code_reader_fn,
    name = "code_reader",
    description = "this tool can read contents of a file and return results"
)