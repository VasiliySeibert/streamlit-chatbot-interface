# Custom loader to process JSON and enforce URL presence
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

# --- RAG State Definition ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --- Vector Store Setup ---
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")


vector_store_path = "vector_store_website_full_v2"
data_file_path = "website_analysis_results.json"
prompt = ChatPromptTemplate.from_template("""
You are the nfdi4ing Copilot. You know all about the personas Alex, Betty, Caden, Doris, Ellen and Fiona. These archetypes represent reoccurring needs of researchers. 
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

# Retrieval step with additional debug logging and filtering by unique URL.
def retrieve(state: State, vector_store):
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
def generate(state, llm):
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

def load_or_create_vectorStore(vector_store_path, embeddings):
    # Load or create vector store based on the selected dataset.
    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded vector store from disk. vector_store_path:", vector_store_path)
    else:
        print("⏳ Generating vector store...")
        docs = custom_load_documents(data_file_path)
        
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
    return vector_store

def process_question(full_question: str, vector_store) -> str:
    """
    Process a question by sequentially performing document retrieval and generation.
    
    Args:
        full_question: The user’s question along with any prior chat history.
    
    Returns:
        The generated answer as a string.
    """
    # Initialize the state with the input question.
    state = {"question": full_question}
    
    # Call the retrieval function to get context documents.
    # Make sure `vector_store` is defined in your module's scope.
    retrieval_result = retrieve(state, vector_store)
    state.update(retrieval_result)
    
    # Call the generation function to generate the answer based on the context.
    # Make sure `llm` is defined in your module's scope.
    generation_result = generate(state, llm)
    state.update(generation_result)
    
    # Return the generated answer.
    return state.get("answer", "")
