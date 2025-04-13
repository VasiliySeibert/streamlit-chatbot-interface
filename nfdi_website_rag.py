import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate

# Setup LLM and Embeddings
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# File paths
local_file_path = "website_analysis_results.json"
vector_store_path = "vector_store_website_full_v2"  # Changed name to force regeneration

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
            # Create a content block with multiple fields to ensure important details are included
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
    
    # Verify that documents have valid URL in metadata
    for doc in docs[:3]:
        print("Document URL:", doc.metadata.get("url"))
    
    # Adjust chunk splitter: reduce chunk size to make key information more prominent.
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

# Use a prompt that instructs the assistant to always cite URLs.
prompt = ChatPromptTemplate.from_template("""
You are an expert assistant helping with research questions. Use the retrieved documents below to answer the question in as much detail as needed.

- Always include source URLs after each relevant point.
- If multiple documents are relevant, combine them.
- If you don't know the answer, say so.

Question: {question}
Context:
{context}

Answer:
""")

# Define state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Revised Retrieval Step with additional logging for debugging
def retrieve(state: State):
    raw_docs = vector_store.similarity_search(state["question"], k=5)
    print(f"[DEBUG] Raw documents retrieved: {len(raw_docs)}")
    for i, doc in enumerate(raw_docs):
        print(f"  Doc {i+1}: URL={doc.metadata.get('url')}, snippet={doc.page_content[:100]}...")
    
    unique_docs = {}
    for doc in raw_docs:
        url = doc.metadata.get("url")
        if url:
            # Ensure each URL appears only once; adjust if multiple chunks per URL are needed.
            if url not in unique_docs:
                unique_docs[url] = doc
        else:
            print("[DEBUG] Document without URL encountered; skipping.")
    
    if not unique_docs:
        print("⚠️ No relevant documents retrieved with valid URLs.")
    return {"context": list(unique_docs.values())}

# Generation Step that groups content by URL for clear citation.
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

    web_sources = [url for url in grouped_docs if url.startswith("http")]
    if web_sources:
        sources_list = "\n\nSources used in this answer:\n" + "\n".join(f"- {url}" for url in web_sources)
        final_answer = response.content.strip() + sources_list
    else:
        final_answer = response.content.strip()

    return {"answer": final_answer}

# Build LangGraph pipeline
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Ask your question using the refined process
response = graph.invoke({
    "question": "What can you tell me about international Benchmarking Studies? Please provide me the URL you derived the information from. Try to include insights from each distinct source at least once."
})
print(response["answer"])
