import os
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from datetime import datetime
from main_llm import LLM
from transformers import pipeline, BertTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate

from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from typing import List
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.


# Define the tools the agent can use
tools = []
 
# 1. Example Custom Tool (gets the current time and location)
def get_current_context():
    """Returns the current time and location."""
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    current_location = "Delhi, Delhi, India"  # As per the context
    return f"The current time is {current_time} in {current_location}."
  

current_context_tool = Tool(
    name="Current Context",
    func=get_current_context,
    description="useful for getting the current time and location.",
)


# 2 No API key is typically needed for DuckDuckGoSearchAPIWrapper
 
def create_duckduckgo_search_tool():
    """Creates a Langchain tool for DuckDuckGo Search, incorporating current context."""
    search = DuckDuckGoSearchAPIWrapper()
 
    def search_with_context(query: str) -> str:
        """Performs a DuckDuckGo search, incorporating the current time and location for better relevance."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z%z")
        context_aware_query = f"{query} (current time: {current_time}, location: Delhi, India)"
        print(f"Performing context-aware DuckDuckGo search for: '{context_aware_query}'")
        return search.run(context_aware_query)
 
    duckduckgo_search_tool = Tool(
        name="DuckDuckGo Search",
        func=search_with_context,
        description="useful for when you need to answer questions about current events or general knowledge using the DuckDuckGo search engine. Input should be a search query. This tool is aware of the current time and location for potentially better results.",
    )
    return duckduckgo_search_tool
 


# 3. Simple Calculation Tool (using Python's eval - be cautious with untrusted input)
def calculate(expression):
    """Evaluates a simple mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
 
calculation_tool = Tool(
    name="Calculator",
    func=calculate,
    description="useful for performing mathematical calculations. Input should be a valid Python expression.",
)

# 4 Example Tool (a simple llm search function)
def web_search(query):
    """Simulates a llm search and returns a snippet."""
    print(f"Simulating llm search for: '{query}'")
    return f"Search results for '{query}': ... relevant information ..."
 
#tools.append(create_duckduckgo_search_tool)
tools.append(current_context_tool)
tools.append(calculation_tool)
#tools.append(web_search)

#Initialize the AI agent
tokenizer_path = "tokenizers/Qwen/Qwen2.5-1.5B-Instruct"
model_path = "models/Qwen/Qwen2.5-1.5B-Instruct"
    
nlp = pipeline("text-generation", model=  model_path, tokenizer = tokenizer_path)
llm = HuggingFacePipeline(pipeline=nlp)



def getResonsefromLLM(prompt):   
    print("=========== LLM called ================")
    llm = LLM()
    result = llm.GetResoponse(prompt)
    return result


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are very powerful assistant, use tools only when its required.
            1. You don't know how to calculate square of a number so you will use method getSquareOfNumber.
            """
        ),
        (
            "user","{input}"
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="chat_history")
    ]
)



my_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  
    prompt=prompt,
    output_key = "result",
    max_iterations=1,
    early_stopping_method="generate",
    memory = ConversationBufferMemory(memory_key = 'chat_history',
        return_messages=True,), 
    handle_parsing_errors="Check you output and make sure it conforms! Do not output an action and a final answer at the same time.",
)

def getResonsefromAgenticLLM(prompt):
    print("=========== Agent called ================")
    # res = my_agent.invoke(({"input": prompt}))
    response= my_agent.run(prompt)
    #res = my_agent.invoke(prompt)["output"]
    return response
    


if __name__ == "__main__":
    # duckduckgo_search = create_duckduckgo_search_tool()
 
    # if duckduckgo_search:
    #     # Example usage of the tool directly (outside of an agent)
    #     query = "latest technology news"
    #     print(f"Searching DuckDuckGo for: '{query}'...")
    #     results = duckduckgo_search.run(query)
    #     print(f"Search Results:\n{results}")
 
    query1 = "What is the current time in our location?"
    print(f"User Query 1: {query1}")
    #print(my_agent.run(query1))
    res = getResonsefromAgenticLLM(query1)
    print("============================================")
    print(res)

 