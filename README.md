# ChatBot + Langchain + Agentic AI, Python Project .... 

---

# Langchain-Python-AgenticAI

This guide walks you through setting up a Python environment, installing dependencies, configuring Pre trained model, and running a chatbot to get response.

Here's an project for chatbot build upon "Qwen/Qwen2.5-1.5B-Instruct" public model using langchain agentic AI. 
Not authentication is required for this model, and read the instructions to run project:
User can select for Agentic AI resposne from UI.
To get agentic AI response use Toggle button.
By default normal LLM respose will be displayed.
Agentic AI is using two tools (Current time tool and Calculator), Check Screenshot
---

## 1. Create a Virtual Environment

Creating a virtual environment helps isolate dependencies and prevents conflicts with other Python projects.

### **For Windows (Command Prompt)**
```sh
python -m venv langchain-env
langchain-env\Scripts\activate
```

### **For macOS/Linux (Terminal)**
```sh
python -m venv langchain-env
source langchain-env/bin/activate
```

---

## 2. Install Requirements

Once the virtual environment is activated, install the required dependencies.

```
pip install -r requirements.txt
```

---

## 3. Configure Model download

Run the following command for downloading qwen model:

```
python qwen_downloader.py 
```
-----

Run the following command for downloading ollama model(access required):

```
python ollama_downloader.py 
```
-----

## 4. Check for Chatbot

Run the following Python command for UI chatbot:

```
streamlit run streamlit_chatbot.py
```

```
To get agentic AI response use Toggle button.
By default normal LLM respose will be displayed.
Check Screenshots
```
-------------


Run the following Python command for chatbot without UI:

```
python qwen_llm_chatbot.py 
```
-------------