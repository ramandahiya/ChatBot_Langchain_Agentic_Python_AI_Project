from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(f"tokenizers/{model_name}") 
model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}")



print("\n🔹 **Initializing chatbot:** 🔹\n")
 
while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    print("\n🔹 **Fetching response:** 🔹\n")
    prompt = question
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=212
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


    print("\n🔹 **Answer:**")
    print(response)
