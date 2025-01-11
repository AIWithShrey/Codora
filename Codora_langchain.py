from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import find_dotenv, load_dotenv
import PIL.Image
import streamlit as st
import os

load_dotenv(find_dotenv())
google_api_key = os.environ['GOOGLE_AI_API_KEY']

prompt = hub.pull("hwchase17/react-chat")

template = """
You are Codora, an AI assistant that helps the user with his/her code. 
You accept user input and provide assistance in coding. Do not provide answers, but guide the user to the right direction.
Provide any necessary information to the user to help them understand the problem and solve it on their own.
Again, do not provide answers, but guide the user to the right direction.
Your answers should be crisp, clear and concise.
Use the provided tool to search Stack Overflow for relevant information.
"""

memory = ChatMessageHistory(session_id="codora")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             google_api_key=google_api_key,
                             temperature=0.3)

tools = load_tools(["stackexchange"])

agent = create_react_agent(llm=model,
                           prompt=prompt,
                           tools=tools)

agent_executor = AgentExecutor(agent=agent,
                               verbose=True,
                               tools=tools)

def agent_response(user_input):
        response = agent_executor.invoke({"input": user_input, "chat_history": memory})
        return response["output"]

st.set_page_config(layout="wide", page_title="Codora AI")

st.title("Codora - Your personal coding mentor")
st.text("""Codora is an AI assistant that helps you with your code. You can ask questions, seek guidance, and get assistance in coding. 
Codora will guide you in understanding and solving your programming issues without giving direct code solutions. 
Let's start coding!""")

st.text("""UPDATE: Codora can now do its own research by searching Stack Overflow for you! 
Just ask your question, and Codora will find the most relevant results for you to explore.""")

messages = [SystemMessage(content=template)]

uploaded_image = st.file_uploader("Upload an image of your code", type=['jpeg', 'png', 'jpg'])
if prompt := st.chat_input("Enter your code here"):
    with st.chat_message("User"):
        st.markdown(prompt)
        messages.append(HumanMessage(content=prompt))
    with st.chat_message("Codora"):
          with st.spinner("Researching and generating response..."):
            try:
                if uploaded_image:
                    image = PIL.Image.open(uploaded_image)
                    response = agent_response([messages, image])
                else:
                    response = agent_response(messages)
                st.write(response)
                messages.append(AIMessage(content=response))
            except KeyError as e:
                st.error(f"An error occurred: {e}. Please try again.")