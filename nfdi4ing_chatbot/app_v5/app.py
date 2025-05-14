import os
import shelve
import streamlit as st
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import List, TypedDict
import utils_website  
import utils_test 
import utils_publications
import utils_arxive
  
import streamlit as st




# --- Setup ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

USER_AVATAR = "👤"
BOT_AVATAR = "https://nfdi4ing.de/wp-content/uploads/2024/09/NFDI4ING_Wort-Bildmarke_NEG_RGB.svg"


# --- Sidebar: Dataset Selection ---
dataset_selection = st.sidebar.radio(
    "Select Data Source", 
    ("Website", "Test", "Betty Copilot", "NFDI4Ing Publications", "ArXiv"),
    key="dataset_selection"
)
st.sidebar.markdown(f"**Current data source:** {dataset_selection}")


# --- Set file paths and prompt based on selection ---
if dataset_selection == "Website":
    print("Loading website data...")
    vector_store_path = utils_website.vector_store_path
    data_file_path = utils_website.data_file_path
    prompt = utils_website.prompt
    embeddings = OpenAIEmbeddings()
    vector_store = utils_website.load_or_create_vectorStore(vector_store_path, embeddings)
    # Define variables for the image URL and title.
    image = "https://nfdi4ing.de/wp-content/uploads/2024/09/NFDI4ING_Wort-Bildmarke_NEG_RGB.svg"
    title = "NFDI4Ing Copilot"

    # --- Streamlit Header ---
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="{image}" width="200"/>
        <h1>{title}</h1>
    </div>
    """, unsafe_allow_html=True)
if dataset_selection == "Test":
    print("Loading publication data...")
    vector_store_path = utils_test.vector_store_path
    data_file_path = utils_test.data_file_path
    prompt = utils_test.prompt
    embeddings = OpenAIEmbeddings()
    vector_store = utils_test.load_or_create_vectorStore(vector_store_path, embeddings)
    image = "https://nfdi4ing.de/wp-content/uploads/2024/09/Betty_mitHintergrund_landscape-2048x1024.jpg"
    title = "Copilot Betty"
    # --- Streamlit Header ---
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="{image}" width="600"/>
        <h1>{title}</h1>
        <h3 style="margin-top: 0.2em; color: gray;">{"Hello, I’m Betty. I’m an engineer and self-taught programmer that develops research software. How can i Help you today?"}</h3>
    </div>
    """, unsafe_allow_html=True)

if dataset_selection == "NFDI4Ing Publications":
    print("Loading publication data...")
    vector_store_path = utils_publications.vector_store_path
    data_file_path = utils_publications.data_file_path
    prompt = utils_publications.prompt
    embeddings = OpenAIEmbeddings()
    vector_store = utils_publications.load_or_create_vectorStore(vector_store_path, embeddings)
    image = "https://nfdi4ing.de/wp-content/uploads/2024/09/NFDI4ING_Wort-Bildmarke_NEG_RGB.svg"
    title = "NFDI4Ing Publications Copilot"
    # --- Streamlit Header ---
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="{image}" width="200"/>
        <h1>{title}</h1>
    </div>
    """, unsafe_allow_html=True)

if dataset_selection == "ArXiv":
    print("Loading publication data...")
    vector_store_path = utils_arxive.vector_store_path
    data_file_path = utils_arxive.data_file_path
    prompt = utils_arxive.prompt
    embeddings = OpenAIEmbeddings()
    vector_store = utils_arxive.load_or_create_vectorStore(vector_store_path, embeddings)
    image = "https://nfdi4ing.de/wp-content/uploads/2024/09/NFDI4ING_Wort-Bildmarke_NEG_RGB.svg"
    title = "Publications test with arxive"
    # --- Streamlit Header ---
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="{image}" width="200"/>
        <h1>{title}</h1>
    </div>
    """, unsafe_allow_html=True)

# --- Chat History Functions ---
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# --- Display Previous Chat Messages ---
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt_input := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # Optionally aggregate recent chat history to provide context.
            history = ""
            for m in st.session_state.messages[-8:]:
                prefix = "User:" if m["role"] == "user" else "Assistant:"
                history += f"{prefix} {m['content']}\n"
            full_question = f"{history}User: {prompt_input}"
            
            # Process the question sequentially.
            if dataset_selection == "Website":
                final_response = utils_website.process_question(full_question, vector_store)
            if dataset_selection == "Test":
                final_response = utils_test.process_question(full_question, vector_store)
            if dataset_selection == "NFDI4Ing Publications":
                final_response = utils_publications.process_question(full_question, vector_store)
            if dataset_selection == "ArXiv":
                final_response = utils_arxive.process_question(full_question, vector_store)
        except Exception as e:
            final_response = f"❌ An error occurred during processing: {str(e)}"

        message_placeholder.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
    save_chat_history(st.session_state.messages)
