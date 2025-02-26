# from llama._index.llms.ollama import Ollama
# from llama_index.llms.ollama import Ollama

from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context, code_parser_template
from code_reader import code_reader
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
import ast


# parse pdf into some structured data
# then convert it to Vectorestoreindex 
# llm will utilise this db and extract just the info it needs to ans a specific query
# vector embeddings take textual data and they embed iinto multi d space that allows us to query based on things like sentiment etc

load_dotenv() # looks for .env file and loads variables and gives access to them
llm = Ollama(model="mistral", request_timeout=30.0)
parser = LlamaParse(result_type="markdown")
# llama parse takes docs and push them to cloud will then be parsed and parsing will be returned to us

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
# takes files and loads and uses appropriate file extractor
# pass these docs to vector store index and create embeddings
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# result = query_engine.query("routes in api?")
# print(result)

# wrapping query engine in a tool that we can provide to ai agent
tools = [QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="api_documentation",
        description="gives docuementation about code for an api"
    ),
),
code_reader
]
# take the result from codellama and write it to a file (since output contains some other stuff also along with code, we need another llm that parses op into format where we can take and write to file)
# LLm number 2 here..codellama basicaly generates code
code_llm=Ollama(model="codellama", request_timeout=1000)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context="context")

class CodeOutput(BaseModel):
    code: str
    descriptions: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_template = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain = [json_prompt_template, llm])

# will take the code_parser_template string and it will inject the end of that string the format of pedantic model (The CodeOutput class)

while(prompt:=input("enter a prompt (q for quit)")) != "q":
    retries = 0
    while retries<3:
        try:
            result = agent.query(prompt)
            # print(result)
            next_result = output_pipeline.run(response=result)
            print(next_result)
            # something like assistant{"code": "...", "description":"...","filename":"..."} gets generated
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant",""))
            print("code generated")
            print(cleaned_json["code"])
            print("\n\nDescription", cleaned_json["description"])
            filename = cleaned_json["filename"]
            break
        except Exception as e:
            retries += 1

# llm = Ollama(model="mistral", request_timeout=30.0)
# # llm = Ollama(
# #     model = "mistral", request_timeout=30.0
# # )
# result = llm.complete("hello")
# print(result)

# LLm number 2 here..codellama basicaly generates code














