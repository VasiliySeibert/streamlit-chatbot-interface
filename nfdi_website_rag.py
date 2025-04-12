from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os
from langchain_core.prompts import ChatPromptTemplate

# Setup LLM and Embeddings
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# File paths
local_file_path = "website_analysis_results.json"
vector_store_path = "vector_store_website_full"

# Load or create vector store
if os.path.exists(vector_store_path):
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ Loaded vector store from disk.")
else:
    print("⏳ Generating vector store...")

    try:
        loader = JSONLoader(
            file_path=local_file_path,
            jq_schema=".[]",
            text_content=False
        )
        docs = loader.load()
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        docs = []

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    all_splits = []
    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        all_splits.extend([
            Document(page_content=chunk, metadata=doc.metadata.copy())
            for chunk in splits
        ])


    # Create and persist vector store
    vector_store = FAISS.from_documents(all_splits, embeddings)
    vector_store.save_local(vector_store_path)
    print("✅ Vector store saved to disk.")

# Load prompt


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

# Retrieval step
def retrieve(state: State):
    raw_docs = vector_store.similarity_search(state["question"], k=15)

    # Filter for unique URLs (if available)
    seen = set()
    diverse_docs = []
    for doc in raw_docs:
        url = doc.metadata.get("url")
        if url and url not in seen:
            seen.add(url)
            diverse_docs.append(doc)
        elif not url:
            # Include non-URL docs only if fewer than 5 found
            if len(diverse_docs) < 5:
                diverse_docs.append(doc)

    if not diverse_docs:
        print("⚠️ No relevant documents retrieved.")

    return {"context": diverse_docs}




# Generation step
# Generation step
def generate(state: State):
    docs = state["context"]

    # Helper to extract the URL reliably
    def get_url(doc):
        return doc.metadata.get("url") or doc.metadata.get("source", "N/A")

    # Group documents by URL
    from collections import defaultdict
    grouped_docs = defaultdict(list)
    for doc in docs:
        url = get_url(doc)
        grouped_docs[url].append(doc.page_content)

    # Build grouped context
    grouped_context = ""
    for url, chunks in grouped_docs.items():
        combined_text = "\n\n".join(chunks)
        grouped_context += f"Context from {url}:\n{combined_text}\n\n"

    # Format the final prompt
    messages = prompt.invoke({
        "question": state["question"],
        "context": grouped_context.strip()
    })

    response = llm.invoke(messages)

    # Clean URL list for citation (only valid web links)
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

# Ask your question
# response = graph.invoke({"question": "Where can I find the PDF paper of the dasdenoising repository?"})
# response = graph.invoke({"question": "What Repositories can help me with denoising raw waveforms?"})
# response = graph.invoke({"question": "What does frank do? Please provide me the URL you derived the information from."})
response = graph.invoke({"question": "What can you tell me about Betty? Please provide me the URL you derived the information from. Try to include insights from each distinct source at least once."})
print(response["answer"])
# print(response)
