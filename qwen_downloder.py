from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

#login(token = 'hf_PwTYtRtroVgzgzvTxGcJzGhDeSRePJdMX')

model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # public LLM model 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

 
tokenizer.save_pretrained(f"tokenizers/{model_name}") 
model.save_pretrained(f"models/{model_name}")
 
