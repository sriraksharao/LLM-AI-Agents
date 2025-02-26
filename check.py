from llama_index.llms.ollama import Ollama
llm = Ollama(model="codellama", request_timeout=60.0)
print(llm.complete("hello"))

# this is just a dummy file to see if the model is working
 