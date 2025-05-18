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
prompt = st.chat_input("Say something or request an image...")

if prompt:
    # Append user message immediately
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Check for image request keyword
    if "generate image:" in prompt.lower():
        image_prompt = prompt.lower().replace("generate image:", "").strip()

        # Generate image using model
        with st.spinner("Generating image..."):
            image_url = llm.invoke_image(image_prompt)  # Assuming the model has an image generation API

        if image_url:
            st.session_state.chat_history.append(AIMessage(content=f"Generated an image for: {image_prompt}\n[IMAGE]{image_url}[/IMAGE]"))
        else:
            st.error("Failed to generate image.")

    else:
        # Text response
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.chat_history)

        # Append AI message after getting the response
        st.session_state.chat_history.append(AIMessage(content=response.content))

# Display the entire chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        if "[IMAGE]" in msg.content:
            content, image_url = msg.content.split("[IMAGE]")[0], msg.content.split("[IMAGE]")[1].split("[/IMAGE]")[0]
            st.chat_message("assistant").write(content)
            st.chat_message("assistant").image(image_url, caption="Generated Image")
        else:
            st.chat_message("assistant").write(msg.content)
