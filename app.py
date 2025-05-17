import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize the chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

st.title("🧠 Gemini Chatbot")

# Chat display
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input
if prompt := st.chat_input("Say something..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Call the model
    with st.spinner("Thinking..."):
        response = llm.invoke(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=response.content))

    # Display AI message
    st.chat_message("assistant").write(response.content)
