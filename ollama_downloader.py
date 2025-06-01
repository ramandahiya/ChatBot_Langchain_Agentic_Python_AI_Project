
import torch
 
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# login(token = 'hf_PwTYtRtroVgzgzvTxGcJzGhDeSRePJdMX')
access_token = 'hf_PwTYtRtroVgzgzvTxGcJzGhDeSRePJdMX'

 
model_name = "meta-llama/Llama-3.2-1B"
 
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token) 
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
 
tokenizer.save_pretrained(f"tokenizers/{model_name}") 
model.save_pretrained(f"models/{model_name}")
 
 
tokenizer = AutoTokenizer.from_pretrained(f"tokenizers/{model_name}") 
model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}")
 
prompt = "What is the capital of India?"
 
prompt_embeddings = tokenizer(prompt, return_tensors="pt")
 
print(prompt_embeddings)
 
response = model.generate(**prompt_embeddings)
 
print(response)
print(tokenizer.decode(response [0])) 