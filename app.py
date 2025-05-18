import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# For Google GenAI SDK (Image Generation)
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os

# Initialize LangChain chat model for text
llm_text = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize Google GenAI client for image generation
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

st.title("🧠 Gemini Chatbot with Image Generation")

# Chat input
prompt = st.chat_input("Say something or request an image...")

if prompt:
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    if "generate image:" in prompt.lower():
        image_prompt = prompt.lower().replace("generate image:", "").strip()
        with st.spinner("Generating image..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=image_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )
                text_response = ""  
                image_url = None

                for part in response.candidates[0].content.parts:
                    if part.text:
                        text_response += part.text
                    elif part.inline_data:
                        image = Image.open(BytesIO(part.inline_data.data))
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        image_url = f"data:image/png;base64,{img_str}"

                st.session_state.chat_history.append(
                    AIMessage(content=text_response + (f"\n[IMAGE]{image_url}[/IMAGE]" if image_url else ""))
                )

            except Exception as e:
                st.error(f"Image generation failed: {str(e)}")

    else:
        with st.spinner("Thinking..."):
            try:
                response = llm_text.invoke(st.session_state.chat_history)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Chat generation failed: {str(e)}")

# Display chat history
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
