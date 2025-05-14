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

vector_store_path = "vector_store_1"
data_file_path = "data.json"
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. You have to answer the following question based on the context provided.
Always say: Surething man at the beginning of your answer.

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
            name = entry.get("name", "")
            profession = entry.get("profession", "")
            fun_fact = entry.get("fun_fact", [])
            
            content = f"name: {name}\nProfession: {profession}\nFun Fact:\n{fun_fact}"
            docs.append(Document(page_content=content, metadata={"name": name}))
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
    return docs

# Retrieval step with debug logging (URL logic removed)
def retrieve(state: State, vector_store):
    raw_docs = vector_store.similarity_search(state["question"], k=5)
    print(f"[DEBUG] Raw documents retrieved: {len(raw_docs)}")
    for i, doc in enumerate(raw_docs):
        print(f"  Doc {i+1}: name={doc.metadata.get('name')}, snippet={doc.page_content[:100]}...")
    return {"context": raw_docs}

# --- Generation Step (with URL references removed) ---
def generate(state, llm):
    docs = state["context"]

    # Use "name" as the identifier for grouping
    def get_identifier(doc):
        return doc.metadata.get("name", "N/A")
    
    from collections import defaultdict
    grouped_docs = defaultdict(list)
    for doc in docs:
        identifier = get_identifier(doc)
        grouped_docs[identifier].append(doc.page_content)
    
    grouped_context = ""
    for identifier, chunks in grouped_docs.items():
        combined_text = "\n\n".join(chunks)
        grouped_context += f"Context from {identifier}:\n{combined_text}\n\n"
    
    messages = prompt.invoke({
        "question": state["question"],
        "context": grouped_context.strip()
    })
    
    response = llm.invoke(messages)
    
    # Simply return the response without attempting to build source citations.
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
        
        # Split texts into chunks to better capture key information.
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
    
    # Retrieve context documents.
    retrieval_result = retrieve(state, vector_store)
    state.update(retrieval_result)
    
    # Generate the answer based on the context.
    generation_result = generate(state, llm)
    state.update(generation_result)
    
    return state.get("answer", "")
