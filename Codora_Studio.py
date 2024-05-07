import PIL.Image
import google.generativeai as genai
import streamlit as st
import time
import PIL
import requests
import re
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))

model_instructions = """
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, guiding the user towards understanding and solving their programming issues without giving direct code solutions. Action can be of three types:

(1) <search>query</search>, which searches Stack Overflow for the query and returns the most relevant results. If the exact answer isn't found, it will suggest relevant topics or threads that the user can explore further.

(2) <lookup>keyword</lookup>, which searches within the found Stack Overflow discussions for the keyword. This function is meant to pinpoint specific information within a larger context, helping to refine the response based on the user's follow-up queries or additional details they might need.

(3) <finish>guidance</finish>, which synthesizes the information gathered through searches and lookups into a coherent piece of guidance. This step concludes the task by providing a clear, concise, and useful suggestion or explanation that guides the user in debugging or understanding their code better, based on best practices and expert discussions from Stack Overflow.

Each step is designed to progressively aid the user in problem-solving by using a systematic approach to extract and interpret useful information from Stack Overflow, encouraging learning and self-reliance in debugging and coding tasks.
"""

examples = """
Here are some examples.

Question
How can I prevent SQL injection in PHP?

Thought 1
To guide the user effectively, I need to find information on preventing SQL injection in PHP using best practices and secure methods.

Action 1
<search>prevent SQL injection PHP</search>

Observation 1
Several methods exist to prevent SQL injection in PHP, including using prepared statements with PDO or MySQLi.

Thought 2
The answer mentions prepared statements, but it's important to show how they are implemented in PHP for better understanding.

Action 2
<lookup>prepared statements PHP example</lookup>

Observation 2
Using prepared statements in PHP can be done using PDO. For example, using the PDO::prepare() method ensures that SQL statements are securely prepared and executed.

Thought 3
I should provide a general explanation and a code example to guide the user comprehensively.

Action 3
<finish>Use PDO for prepared statements to prevent SQL injection. For example, \\{{stmt = \\{{pdo->prepare('SELECT * FROM users WHERE email = :email'); \\{{stmt->execute(['email' => \\{{email]);}}</finish>

Question
What is the difference between INNER JOIN and LEFT JOIN in SQL?

Thought 1
I need to explain the difference between INNER JOIN and LEFT JOIN with clear definitions and examples.

Action 1
<search>INNER JOIN vs LEFT JOIN SQL</search>

Observation 1
INNER JOIN returns rows when there is a match in both tables, whereas LEFT JOIN returns all rows from the left table, and the matched rows from the right table, plus NULLs if there is no match.

Thought 2
To make it clear, I should provide a SQL example showing the use of both types of joins.

Action 2
<finish>INNER JOIN is used when you want to return only the rows where there is a match in both joined tables. LEFT JOIN returns all rows from the left table, and the matching rows from the right table, or NULL if there is no match. For example, SELECT * FROM orders INNER JOIN customers ON orders.customer_id = customers.id;</finish>

Question
Why is my JavaScript fetch function not working?

Thought 1
There could be several reasons why a fetch function might not work; I should find the most common issues and solutions.

Action 1
<search>JavaScript fetch function not working</search>

Observation 1
Common issues with the fetch function can include CORS errors, network issues, or incorrect headers.

Thought 2
Since CORS is a frequent issue, I should explain what it is and how to solve it.

Action 2
<lookup>CORS solution</lookup>

Observation 2
Cross-Origin Resource Sharing (CORS) issues can be resolved by ensuring the server includes the appropriate CORS headers.

Thought 3
I need to provide specific guidance on setting headers in both client-side and server-side code.

Action 3
<finish>Ensure your server is set to accept requests from your domain. Add headers like 'Access-Control-Allow-Origin: *' in server responses. In your fetch request, check your URL and response handling.</finish>

Question
How do I use async/await in a JavaScript function?

Thought 1
The user needs to understand the syntax and basic usage of async/await in JavaScript.

Action 1
<search>using async await JavaScript</search>

Observation 1
The async keyword is added to functions to tell them to return a promise rather than directly returning the value. Await is used to pause the code on that line until the promise fulfills, then returns the resulting value.

Thought 2
I should provide a simple code example to illustrate the use of async/await.

Action 2
<finish>To use async/await, declare your function with async and use await inside it. For example: async function getUser() {{ let response = await fetch('https://api.user.com'); let data = await response.json(); return data; }}</finish>

Question
{question}
"""

ReAct_prompt = model_instructions + examples
with open('model_instructions.txt', 'w') as f:
  f.write(ReAct_prompt)

class ReAct:
  def __init__(self, model: str, 
               ReAct_prompt: str | os.PathLike, 
               system_instruction: str):
    """Prepares Gemini to follow a `Few-shot ReAct prompt` by imitating
    `function calling` technique to generate both reasoning traces and
    task-specific actions in an interleaved manner.

    Args:
        model: name to the model.
        ReAct_prompt: ReAct prompt OR path to the ReAct prompt.
    """
    self.model = genai.GenerativeModel(model,
                                       system_instruction=system_instruction)
    self.chat = self.model.start_chat(history=[])
    self.should_continue_prompting = True
    self._search_history: list[str] = []
    self._search_urls: list[str] = []

    try:
      # try to read the file
      with open(ReAct_prompt, 'r') as f:
        self._prompt = f.read()
    except FileNotFoundError:
      # assume that the parameter represents prompt itself rather than path to the prompt file.
      self._prompt = ReAct_prompt

  @property
  def prompt(self):
    return self._prompt

  @classmethod
  def add_method(cls, func):
    setattr(cls, func.__name__, func)

  @staticmethod
  def clean(text: str):
    """Helper function for responses."""
    text = text.replace("\n", " ")
    return text

@ReAct.add_method
def search_stack_overflow(self, query):
    """Search Stack Overflow for a given query and return the results.

    Args:
        query (str): The search keyword or phrase.

    Returns:
        str: Formatted string listing the top questions or an error message.
    """
    url = "https://api.stackexchange.com/2.2/search/advanced"
    params = {
        'order': 'desc',
        'sort': 'relevance',
        'q': query,
        'site': 'stackoverflow',
        'key': os.getenv("STACK_OVERFLOW_API_KEY")  # Replace with your actual API key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get('items', [])
        if not results:
            return "No results found."
        formatted_results = "\n".join(f"{idx + 1}. {item['title']} - {item['link']}" for idx, item in enumerate(results))
        return formatted_results
    except requests.exceptions.RequestException as e:
        return f"Failed to search Stack Overflow: {str(e)}"
    
@ReAct.add_method
def search(self, query):
    """
    Conducts a search on Stack Overflow and formats the response for further actions.

    Args:
        query (str): Search query.

    Returns:
        str: Formatted string listing the top questions.
    """
    results = self.search_stack_overflow(query)
    if isinstance(results, str):  # Handling no results or errors.
        return results
    formatted_results = "\n".join(f"{idx + 1}. {item['title']} - {item['link']}" for idx, item in enumerate(results))
    return formatted_results

@ReAct.add_method
def lookup(self, phrase: str, context_length=200):
    """
    Searches for the `phrase` in the content of the most recently retrieved Stack Overflow question or answer 
    and returns the context around the phrase, controlled by the `context_length` parameter.
    
    Args:
        phrase (str): Lookup phrase to search for within the retrieved content.
        context_length (int): Number of characters (not words) to consider around the phrase for context.

    Returns:
        str: Context related to the `phrase` within the retrieved content, including some preceding and following text.
    """
    if not self._search_history:
        return "No recent search history available."

    # Assuming the last search fetched a text content, stored in self._search_content
    content = self._search_content
    # Clean and prepare the content for search
    content = self.clean(content)
    start_index = content.find(phrase)

    if start_index == -1:
        return "Phrase not found in the current context."

    # Extract context around the found phrase
    start = max(0, start_index - context_length)
    end = start_index + len(phrase) + context_length
    result = content[start:end]

    return result

@ReAct.add_method
def finish(self, guidance):
    """Finishes the conversation and provides final guidance."""
    self.should_continue_prompting = False
    return guidance  # Returning the guidance directly

@ReAct.add_method
def __call__(self, user_question, image=None, max_calls=8, **generation_kwargs):
    assert 0 < max_calls <= 8, "max_calls must be between 1 and 8"
    
    if len(self.chat.history) == 0:
        model_prompt = self.prompt.format(question=user_question)
        if image:
            image = PIL.Image.open(image)
            model_prompt = [model_prompt, image]
    else:
        model_prompt = user_question
        if image:
            image = PIL.Image.open(image)
            model_prompt = [model_prompt, image]

    callable_entities = ['</search>', '</lookup>', '</finish>']
    generation_kwargs.update({'stop_sequences': callable_entities})

    self.should_continue_prompting = True
    for idx in range(max_calls):
        if not self.should_continue_prompting:
            break

        response = self.chat.send_message(content=model_prompt,
                                          generation_config=generation_kwargs, stream=False)

        response_cmd = self.chat.history[-1].parts[-1].text

        try:
            cmd_match = re.search(r'<(.*?)>', response_cmd)
            if cmd_match:
                cmd = cmd_match.group(1)
                query = response_cmd.split(f'<{cmd}>')[-1].split(f'</{cmd}>')[0]
                observation = getattr(self, cmd)(query)

                if cmd == 'finish':
                    for word in observation.split():
                        yield word + ' '
                        time.sleep(0.05)
                    #print(f"\nFinal Guidance: {observation}")  # Printing final guidance
                    break

                stream_message = f"\nObservation {idx + 1}\n{observation}"
                print(stream_message)
                model_prompt = f"<{cmd}>{query}</{cmd}>'s Output: {stream_message}"
            else:
                raise ValueError("Command not found in the response.")

        except (AttributeError, ValueError) as e:
            print(f"Error: {str(e)}")
            model_prompt = "Error in processing the response; please check the command format."
            break

generation_config = {
        "max_output_tokens": 2048,
        "temperature": 0.3,
        }

system_instruction = """You are Codora, an AI assistant that helps you with your code. 
You accept user input and provide assistance in coding. Do not provide answers, but guide the user to the right direction.
Provide any necessary information to the user to help them understand the problem and solve it on their own.
Again, do not provide answers, but guide the user to the right direction.
Your answers should be crisp, clear and concise."""

gemini_ReAct_chat = ReAct(model='gemini-1.5-pro-latest', 
                          system_instruction=system_instruction, 
                          ReAct_prompt='model_instructions.txt')


# Streamlit app
st.set_page_config(layout="wide", page_title="Codora AI")

st.title("Codora - Your personal coding mentor")
st.text("Codora is an AI assistant that helps you with your code. You can ask questions, seek guidance, and get assistance in coding. Codora will guide you in understanding and solving your programming issues without giving direct code solutions. Let's start coding!")
st.text("UPDATE: Codora can now do its own research by searching Stack Overflow for you! Just ask your question, and Codora will find the most relevant results for you to explore.")


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
        with st.spinner("Researching and generating response..."):
            stream = gemini_ReAct_chat(prompt, 
                                       uploaded_image,
                                       8 )
            # Display assistant response in chat message container
            response = st.write_stream(stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})