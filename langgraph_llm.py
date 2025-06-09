from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline



# Use a pipeline as a high-level helper
# from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
llm = HuggingFacePipeline(pipeline=pipe)


 
# 1. Define the Graph State
# This is the state that will be passed between nodes.
# It's crucial to define what data your nodes will operate on and update.
class AgentState(TypedDict):
    messages: Annotated[List[str], add_messages]
    # You can add other state variables as needed
    user_query: str
    tool_results: List[str]
    next_node: str # Used for conditional routing
 
# 2. Define the Nodes
# Each node is a Python function that takes the current state as input
# and returns a dictionary of updates to the state.
 
def node_user_query_processor(state: AgentState):
    """
    Processes the initial user query.
    """
    print("---NODE: User Query Processor---")
    user_query = state["user_query"]
    # Simulate some processing of the user query
    processed_query = f"Processed: {user_query.upper()}"
    print(f"Processed Query: {processed_query}")
    # Decide the next node based on some logic (e.g., presence of keywords)
    if "tool" in user_query.lower():
        next_node = "tool_executor"
    else:
        next_node = "llm_responder"
    return {"messages": [f"Query processed: {processed_query}"], "next_node": next_node}
 
def node_tool_executor(state: AgentState):
    """
    Executes a simulated tool.
    """
    print("---NODE: Tool Executor---")
    query = state["user_query"]
    # Simulate a tool call based on the query
    tool_output = f"Tool executed for: '{query}' with result: Data XYZ"
    print(f"Tool Output: {tool_output}")
    return {"messages": [f"Tool output: {tool_output}"], "tool_results": [tool_output]}
 
def node_llm_responder(state: AgentState):
    """
    Generates a response using an LLM (simulated here).
    """
    print("---NODE: LLM Responder---")
    messages = state["messages"]
    tool_results = state.get("tool_results", [])
    
    response_parts = []
    if messages:
        response_parts.append(f"LLM received messages: {messages[-1]}")
    if tool_results:
        response_parts.append(f"LLM considered tool results: {tool_results[-1]}")
 
    llm_response = "LLM generated response: " + " ".join(response_parts)
    print(f"LLM Response: {llm_response}")
    return {"messages": [llm_response]}
 
# 3. Define the Conditional Edge (Router)
def router(state: AgentState) -> str:
    """
    This function determines the next node based on the 'next_node' field in the state.
    """
    print(f"---ROUTER: Deciding next node. Current next_node: {state['next_node']}---")
    if state["next_node"] == "tool_executor":
        return "tool_executor"
    elif state["next_node"] == "llm_responder":
        return "llm_responder"
    else:
        # Fallback or error handling
        return END # Or another specific node
 
# 4. Construct the Graph
workflow = StateGraph(AgentState)
 
# Add nodes to the graph
workflow.add_node("user_query_processor", node_user_query_processor)
workflow.add_node("tool_executor", node_tool_executor)
workflow.add_node("llm_responder", node_llm_responder)
 
# Set the entry point (where the graph starts)
workflow.set_entry_point("user_query_processor")
 
# Add edges:
# Simple edge: from node_user_query_processor to the router
workflow.add_edge("user_query_processor", "router")
 
# Conditional edges: The router determines the next step
workflow.add_conditional_edges(
    "user_query_processor",          # Source node (the router)
    router,            # The function that decides the next node
    {                  # Mapping from function output to next node
        "tool_executor": "tool_executor",
        "llm_responder": "llm_responder",
    },
)
 
# Connect the tool_executor and llm_responder to the END of the graph
workflow.add_edge("tool_executor", END)
workflow.add_edge("llm_responder", END)
 
 
# 5. Compile the Graph
app = workflow.compile()
 
# 6. Invoke the Graph
print("\n---RUN 1: Query requiring tool use---")
final_state_1 = app.invoke({"user_query": "Please search for data using a tool."})
print(f"\nFinal State 1: {final_state_1}")
 
print("\n---RUN 2: Query for direct LLM response---")
final_state_2 = app.invoke({"user_query": "Tell me a joke."})
print(f"\nFinal State 2: {final_state_2}")
 
print("\n---RUN 3: Another query requiring tool use---")
final_state_3 = app.invoke({"user_query": "I need to run a tool for analysis."})
print(f"\nFinal State 3: {final_state_3}")
 