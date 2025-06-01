from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

# Load Hugging Face Summarization Pipeline
model = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# Wrap it inside LangChain
llm = HuggingFacePipeline(pipeline=model)

# Create the prompt template for summarization
template = PromptTemplate.from_template(
    "Summarize the following text in a way a {age} year old would understand:\n\n{text}"
)

summarizer_chain = template | llm

# Get user input
text_to_summarize = input("\nEnter text to summarize:\n")
age = input("Enter target age for simplification:\n")

# Execute the summarization chain
summary = summarizer_chain.invoke({"text": text_to_summarize, "age": age})

print("\n🔹 **Generated Summary:**")
print(summary)
