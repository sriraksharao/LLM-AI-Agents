# from llama._index.llms.ollama import Ollama
# from llama_index.llms.ollama import Ollama

from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

# parse pdf into some structured data
# then convert it to Vectorestoreindex 
# llm will utilise this db and extract just the info it needs to ans a specific query
# vector embeddings take textual data and they embed iinto multi d space that allows us to query based on things like sentiment etc

parser = LlamaParse(result_type="markdown")
# llama parse takes docs and push them to cloud will then be parsed and parsing will be returned to us

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
# takes files and loads and uses appropriate file extractor
# pass these docs to vector store index and create embeddings
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

result = query_engine.query("routes in api?")
print(result)


# llm = Ollama(model="mistral", request_timeout=30.0)
# # llm = Ollama(
# #     model = "mistral", request_timeout=30.0
# # )
# result = llm.complete("hello")
# print(result)