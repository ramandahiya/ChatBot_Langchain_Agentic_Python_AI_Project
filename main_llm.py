from transformers import AutoModelForCausalLM, AutoTokenizer

 
class Singleton:
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # If no instance exists, create a new one
            cls._instance = super().__new__(cls)
            # You can add any initialization logic here if needed
            # For example, if your LLM client needs to load models or connect to an API:
            # cls._instance.llm_client = load_llm_model() 
        return cls._instance
    
class LLM(Singleton):

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = None
    model = None
    tokenizer_path = "tokenizers/Qwen/Qwen2.5-1.5B-Instruct"
    model_path = "models/Qwen/Qwen2.5-1.5B-Instruct"
 
    def __init__(self):   
        print("\n🔹 **Initializing chatbot:** 🔹\n")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.tokenizer_path}") 
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_path}")


    # Instance
    def GetResoponse(self, prompt):
        if self.model is None:
           return ("Download the Qwen model. Run qwen_downloder.py")

        #question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
   
        print("\n🔹 **Fetching response:** 🔹\n")
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=212
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("\n🔹 **Answer:**")
        print(response)
        return response



    

 
   