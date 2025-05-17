import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize the chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

st.title("🧠 Gemini Chatbot")

# Chat input
prompt = st.chat_input("Say something...")

if prompt:
    # Append user message immediately
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Call the model and wait for the response
    with st.spinner("Thinking..."):
        response = llm.invoke(st.session_state.chat_history)
    
    # Append AI message after getting the response
    st.session_state.chat_history.append(AIMessage(content=response.content))

# Display the entire chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
