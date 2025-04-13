import os
import json
import streamlit as st
import shelve
from collections import defaultdict
from dotenv import load_dotenv
from typing_extensions import List, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# Import RapidFuzz for fuzzy matching
from rapidfuzz import process, fuzz

# --- RAG State Definition ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --- Vector Store and LLM Setup ---
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

vector_store_path = "vector_store_nfdi4ingPublications"
data_file_path = "nfdi4ing_works_with_abstracts_partial.json"

prompt = ChatPromptTemplate.from_template("""
You are a research assistant. You have to answer the following research question based on the context provided.
The context consists of multiple research publications. Each publication includes the following details:
- Title
- Authors
- Measure (e.g., the publication venue)
- Year
- Link
- Abstract

Please select the five most relevant publications for addressing the question.
For each selected publication, clearly list:
  • Title
  • Authors
  • Measure
  • Year
  • Link
  • A brief summary of the publication (using the abstract provided)
Finally, synthesize these insights into a comprehensive answer to the question.

Question: {question}
Context:
{context}

Answer:
""")

def custom_load_documents(file_path: str) -> List[Document]:
    """Load documents from a JSON file, extending metadata to include title, year, link, measure, and authors."""
    docs = []
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            title    = entry.get("title", "")
            year     = entry.get("year", "")
            link     = entry.get("link", "")
            measure  = entry.get("measure", "")
            authors  = entry.get("authors", "")
            abstract = entry.get("abstract", "")
            
            # Create the document content using all available fields.
            content = (f"Title: {title}\n"
                       f"Year: {year}\n"
                       f"Link: {link}\n"
                       f"Measure: {measure}\n"
                       f"Authors: {authors}\n"
                       f"Abstract: {abstract}")
            
            metadata = {
                "title": title,
                "year": year,
                "link": link,
                "measure": measure,
                "authors": authors,
            }
            docs.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
    return docs

def build_metadata_index(docs: List[Document], field: str) -> dict:
    """
    Build an inverted index for a specific metadata field.
    For authors, split the authors string (assuming comma separation) and
    build a mapping from each normalized author token to the documents that contain it.
    """
    index = defaultdict(list)
    for doc in docs:
        metadata_value = doc.metadata.get(field, "")
        if not isinstance(metadata_value, str):
            metadata_value = str(metadata_value)
        if metadata_value:
            tokens = [token.strip().lower() for token in metadata_value.split(",")]
            for token in tokens:
                if token:
                    index[token].append(doc)
    return index

def fuzzy_metadata_search(query: str, metadata_index: dict, field: str, score_threshold=80) -> List[Document]:
    """
    Use fuzzy matching to search the metadata index for the query.
    Returns documents for which the fuzzy match score exceeds the threshold.
    """
    query = query.lower()
    results = []
    keys = list(metadata_index.keys())
    # Extract keys similar to the query.
    matches = process.extract(query, keys, scorer=fuzz.partial_ratio, score_cutoff=score_threshold)
    for match, score, _ in matches:
        results.extend(metadata_index[match])
    return results

# Build the metadata index on load (for authors)
all_docs = custom_load_documents(data_file_path)
authors_index = build_metadata_index(all_docs, "authors")

def unique_docs(docs: List[Document]) -> List[Document]:
    """
    Deduplicate documents using a fingerprint created from the page content and sorted metadata.
    """
    unique = []
    seen = set()
    import json
    for doc in docs:
        fingerprint = (doc.page_content, json.dumps(doc.metadata, sort_keys=True))
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(doc)
    return unique

def retrieve(state: State, vector_store, metadata_index: dict = None) -> dict:
    # Perform a vector search.
    vector_docs = vector_store.similarity_search(state["question"], k=5)
    print(f"[DEBUG] Vector search retrieved: {len(vector_docs)} documents")
    
    # Perform fuzzy metadata search (for author matches) regardless of keywords.
    fuzzy_docs = fuzzy_metadata_search(state["question"], metadata_index, "authors")
    print(f"[DEBUG] Fuzzy metadata search retrieved: {len(fuzzy_docs)} documents")
    
    # Prioritize fuzzy matches by putting them at the front.
    fuzzy_set = set(id(doc) for doc in fuzzy_docs)
    # Create a final list starting with the fuzzy results.
    final_docs = list(unique_docs(fuzzy_docs))
    # Then add any vector docs not already in the fuzzy set.
    for doc in vector_docs:
        if id(doc) not in fuzzy_set:
            final_docs.append(doc)
    
    final_docs = unique_docs(final_docs)
    
    for i, doc in enumerate(final_docs):
        title = doc.metadata.get("title", "N/A")
        authors = doc.metadata.get("authors", "N/A")
        print(f"  Doc {i+1}: title={title}, authors={authors}")
    
    return {"context": final_docs}

# --- Generation Step (presenting five most relevant results with prominent authors) ---
def generate(state, llm):
    docs = state["context"]
    top_docs = docs[:5] if len(docs) >= 5 else docs
    context_text = ""
    for i, doc in enumerate(top_docs):
        title   = doc.metadata.get("title", "N/A")
        authors = doc.metadata.get("authors", "N/A")
        measure = doc.metadata.get("measure", "N/A")
        year    = doc.metadata.get("year", "N/A")
        link    = doc.metadata.get("link", "N/A")
        abstract_snippet = doc.page_content[:500]
        context_text += f"Result {i+1}:\n"
        context_text += f"Title: {title}\n"
        context_text += f"Authors: {authors}\n"
        context_text += f"**Prominent Authors: {authors}**\n"
        context_text += f"Measure: {measure}\n"
        context_text += f"Year: {year}\n"
        context_text += f"Link: {link}\n"
        context_text += f"Abstract: {abstract_snippet}\n\n"

    messages = prompt.invoke({
        "question": state["question"],
        "context": context_text.strip()
    })
    
    response = llm.invoke(messages)
    final_answer = response.content.strip()
    return {"answer": final_answer}

def load_or_create_vectorStore(vector_store_path, embeddings):
    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded vector store from disk. vector_store_path:", vector_store_path)
    else:
        print("⏳ Generating vector store...")
        docs = all_docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = []
        for doc in docs:
            splits = text_splitter.split_text(doc.page_content)
            all_splits.extend([Document(page_content=chunk, metadata=doc.metadata.copy()) for chunk in splits])
        vector_store = FAISS.from_documents(all_splits, embeddings)
        vector_store.save_local(vector_store_path)
        print("✅ Vector store saved to disk.")
    return vector_store

def process_question(full_question: str, vector_store) -> str:
    state = {"question": full_question}
    retrieval_result = retrieve(state, vector_store, metadata_index=authors_index)
    state.update(retrieval_result)
    generation_result = generate(state, llm)
    state.update(generation_result)
    return state.get("answer", "")

if __name__ == "__main__":
    vector_store = load_or_create_vectorStore(vector_store_path, embeddings)
    question = st.text_input("Enter your research question:")
    if question:
        answer = process_question(question, vector_store)
        st.write(answer)
