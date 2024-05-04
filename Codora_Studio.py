import PIL.Image
import google.generativeai as genai
import streamlit as st
import time
import PIL
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))

generation_config = {
        "max_output_tokens": 2048,
        "temperature": 0.3,
        }

system_instruction = """You are Codora, an AI assistant that helps you with your code. 
You accept user input and provide assistance in coding. Do not provide answers, but guide the user to the right direction.
Provide any necessary information to the user to help them understand the problem and solve it on their own.
Again, do not provide answers, but guide the user to the right direction.
Your answers should be crisp, clear and concise."""

model = genai.GenerativeModel('gemini-1.5-pro-latest',
                              generation_config=generation_config,
                              system_instruction=system_instruction)

chat = model.start_chat(history=[])

def response_generator(prompt, image=None):
    if image:
        image = PIL.Image.open(image)
        response = chat.send_message([prompt, image],
                                     stream=True)
    else:
        response = chat.send_message(prompt,
                                     stream=True)
    
    for chunk in response:
        for word in chunk.text.split():
            yield word + " "
            time.sleep(0.05)


# Streamlit app

st.title("Codora - Your personal coding mentor")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_image = st.file_uploader("Upload an image of your code", type=['jpeg', 'png', 'jpg'])

# Accept user input
if prompt := st.chat_input("Enter your code here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = response_generator(prompt, uploaded_image)
        # Display assistant response in chat message container
        response = st.write_stream(stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})