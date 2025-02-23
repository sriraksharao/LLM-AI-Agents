# from llama._index.llms.ollama import Ollama
# from llama_index.llms.ollama import Ollama

from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

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
tools = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="api_documentation",
        description="gives docuementation about code for an api"
    ),
)

code_llm=Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context="")

# llm = Ollama(model="mistral", request_timeout=30.0)
# # llm = Ollama(
# #     model = "mistral", request_timeout=30.0
# # )
# result = llm.complete("hello")
# print(result)

# LLm number 2 here..codellama basicaly generates code
code_llm=Ollama(model="codellama")













