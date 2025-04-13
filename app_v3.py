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
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# --- Setup ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

USER_AVATAR = "👤"
BOT_AVATAR = "https://nfdi4ing.de/wp-content/uploads/2024/09/NFDI4ING_Wort-Bildmarke_NEG_RGB.svg"


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

# File paths
local_file_path = "website_analysis_results.json"
vector_store_path = "vector_store_website_full_v2"  # New name to force regeneration

# Custom loader to process JSON and enforce URL presence
def custom_load_documents(file_path: str) -> List[Document]:
    docs = []
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            url = entry.get("url")
            if not url:
                print("⚠️ Skipping entry with missing URL")
                continue
            # Concatenate title, summary, and key takeaways to build a content block.
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            takeaways = entry.get("key_takeaways", [])
            takeaway_text = "\n".join(
                f"- {item.get('title', '')}: {item.get('detail', '')}"
                for item in takeaways
            )
            content = f"Title: {title}\nSummary: {summary}\nKey Takeaways:\n{takeaway_text}"
            docs.append(Document(page_content=content, metadata={"url": url}))
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
    return docs

# Load or create vector store
if os.path.exists(vector_store_path):
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ Loaded vector store from disk.")
else:
    print("⏳ Generating vector store...")
    docs = custom_load_documents(local_file_path)
    
    # Adjust chunk splitter: reduce chunk size to isolate key info.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = []
    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        all_splits.extend([
            Document(page_content=chunk, metadata=doc.metadata.copy())
            for chunk in splits
        ])
    
    vector_store = FAISS.from_documents(all_splits, embeddings)
    vector_store.save_local(vector_store_path)
    print("✅ Vector store saved to disk.")

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_template("""
You are the nfdi4ing Copilot. You know all about the personas Alex, Betty, Caden, Doris, Ellen and Fiona. You know that these archetypes represent reoccuring needs of researchers. 
You have a special focus on the services and offerings of the NFDI4Ing.
Use the retrieved documents below to answer the question in as much detail as possible.

- Always include source URLs after each relevant point.
- If multiple documents are relevant, combine them.
- If you don't know the answer, say so.

Question: {question}
Context:
{context}

Answer:
""")

# --- RAG State Definition ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Retrieval step with additional debug logging and filtering by unique URL
def retrieve(state: State):
    raw_docs = vector_store.similarity_search(state["question"], k=5)
    print(f"[DEBUG] Raw documents retrieved: {len(raw_docs)}")
    for i, doc in enumerate(raw_docs):
        print(f"  Doc {i+1}: URL={doc.metadata.get('url')}, snippet={doc.page_content[:100]}...")
    
    unique_docs = {}
    for doc in raw_docs:
        url = doc.metadata.get("url")
        if url:
            if url not in unique_docs:
                unique_docs[url] = doc
        else:
            print("[DEBUG] Document without URL encountered; skipping.")
    
    if not unique_docs:
        print("⚠️ No relevant documents retrieved with valid URLs.")
    return {"context": list(unique_docs.values())}

# --- Generation Step ---
def generate(state: State):
    docs = state["context"]

    def get_url(doc):
        return doc.metadata.get("url") or doc.metadata.get("source", "N/A")
    
    from collections import defaultdict
    grouped_docs = defaultdict(list)
    for doc in docs:
        url = get_url(doc)
        grouped_docs[url].append(doc.page_content)
    
    grouped_context = ""
    for url, chunks in grouped_docs.items():
        combined_text = "\n\n".join(chunks)
        grouped_context += f"Context from {url}:\n{combined_text}\n\n"
    
    messages = prompt.invoke({
        "question": state["question"],
        "context": grouped_context.strip()
    })
    
    response = llm.invoke(messages)
    
    # Build the final answer with source citations.
    web_sources = [url for url in grouped_docs if url.startswith("http")]
    if web_sources:
        sources_list = "\n\nSources used in this answer:\n" + "\n".join(f"- {url}" for url in web_sources)
        final_answer = response.content.strip() + sources_list
    else:
        final_answer = response.content.strip()
    
    return {"answer": final_answer}

# --- Build Graph ---
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
            # Aggregate the recent chat history for context.
            history = ""
            for m in st.session_state.messages[-8:]:
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
