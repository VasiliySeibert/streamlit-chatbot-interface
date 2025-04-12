from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os

# Setup LLM and Embeddings
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# File paths and vector store directories
files = [
    ("BettysResult_seismology_doi_in_readme.json", "vector_store_1", "first JSON file"),
    ("data.json", "vector_store_2", "second JSON file")
]

vector_stores = {}

# Load or create each vector store
for file_path, store_path, source_tag in files:
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ Loaded vector store for {file_path}")
    else:
        print(f"⏳ Generating vector store for {file_path}")
        try:
            loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = source_tag
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            docs = []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(all_splits, embeddings)
        vector_store.save_local(store_path)
        print(f"✅ Vector store saved for {file_path}")
    
    vector_stores[source_tag] = vector_store

# Load prompt
prompt = hub.pull("rlm/rag-prompt")

# Define state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Retrieval step: collect from both stores
def retrieve(state: State):
    all_docs = []
    for source_tag, store in vector_stores.items():
        retrieved = store.similarity_search(state["question"])
        for doc in retrieved:
            doc.metadata["source"] = source_tag
        all_docs.extend(retrieved)
    return {"context": all_docs}

# Generation step: group by source
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

# Build LangGraph pipeline
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Ask your question
response = graph.invoke({
    "question": "Which Repositories can help me with denoising raw waveforms. Please give me at least three. also who is vasiliy?"
})
print(response["answer"])
