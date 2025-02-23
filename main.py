# from llama._index.llms.ollama import Ollama
# from llama_index.llms.ollama import Ollama

from llama_index.llms.ollama import Ollama

llm = Ollama(model="mistral", request_timeout=30.0)
# llm = Ollama(
#     model = "mistral", request_timeout=30.0
# )
result = llm.complete("hello")
print(result)