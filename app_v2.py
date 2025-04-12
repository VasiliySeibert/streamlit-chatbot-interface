import os
import shelve
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# --- Setup ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# --- Streamlit Header ---
st.markdown("""
<div style="display: flex; flex-direction: column; align-items: center;">
    <img src="https://nfdi4ing.de/wp-content/uploads/2024/09/NFDI4ING_Wort-Bildmarke_NEG_RGB.svg" width="200"/>
    <h1>NFDI4Ing Copilot</h1>
</div>
""", unsafe_allow_html=True)

# --- Vector Store Setup ---
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

files = [
    ("website_analysis_results.json", "vector_store_nfdiWebsite", "NFDI4ING Website:"),
    ("data.json", "vector_store_2", "second JSON file")
]

vector_stores = {}

for file_path, store_path, source_tag in files:
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        try:
            loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = source_tag
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = splitter.split_documents(docs)
            vector_store = FAISS.from_documents(all_splits, embeddings)
            vector_store.save_local(store_path)
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            vector_store = None
    if vector_store:
        vector_stores[source_tag] = vector_store

# --- Prompt & Pipeline ---
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    all_docs = []
    for source_tag, store in vector_stores.items():
        retrieved = store.similarity_search(state["question"])
        for doc in retrieved:
            doc.metadata["source"] = source_tag
        all_docs.extend(retrieved)
    return {"context": all_docs}

def generate(state: State):
    grouped = {}
    for doc in state["context"]:
        source = doc.metadata.get("source", "unknown source")
        grouped.setdefault(source, []).append(doc.page_content)

    answer_parts = []
    for source, docs in grouped.items():
        context = "\n\n".join(docs)
        messages = prompt.invoke({"question": state["question"], "context": context})
        response = llm.invoke(messages)
        answer_parts.append(f"📁 Based on the {source}:\n{response.content}")

    return {"answer": "\n\n".join(answer_parts)}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# --- Chat History ---
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
    if st.button("Chat with Betty"):
        st.session_state.messages = []
        save_chat_history([])

# --- Display Messages ---
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# --- Handle Input ---
if prompt_input := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # 🔁 Include the last 4 chat turns (user and assistant) as context
            history = ""
            for m in st.session_state.messages[-8:]:  # 4 user-assistant pairs = 8 messages
                prefix = "User:" if m["role"] == "user" else "Assistant:"
                history += f"{prefix} {m['content']}\n"
            full_question = f"{history}User: {prompt_input}"

            result = graph.invoke({"question": full_question})
            final_response = result["answer"]
        except Exception as e:
            final_response = f"❌ An error occurred during processing: {str(e)}"

        message_placeholder.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
    save_chat_history(st.session_state.messages)

