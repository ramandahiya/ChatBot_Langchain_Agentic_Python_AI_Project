import streamlit as st
from main_llm import LLM
from langchain_agent import getResonsefromAgenticLLM
from langchain_agent import getResonsefromLLM
from langchain_agent import my_agent

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class Response:
        
    def getResonsefromLLM(prompt):   
        llm = LLM()
        result = llm.GetResoponse(prompt)
        return result


    def getResonsefromAgenticLLM(prompt):   
        result = my_agent.run(prompt)
        return result
        




st.title("Chatbot")


# Create a toggle button
toggle_state = st.toggle("Enable AI Agent")



# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("Act like an guide"))

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("How are you?")

# did the user submit a prompt?
if prompt:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # create the echo (response) and add it to the screen
    if toggle_state:
        result = getResonsefromAgenticLLM(prompt)
    else:
        result = getResonsefromLLM(prompt)

    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))

