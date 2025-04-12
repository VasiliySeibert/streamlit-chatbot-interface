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

# File paths
local_file_path = "BettysResult_seismology_doi_in_readme.json"
vector_store_path = "vector_store_faiss"

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Create and persist vector store
    vector_store = FAISS.from_documents(all_splits, embeddings)
    vector_store.save_local(vector_store_path)
    print("✅ Vector store saved to disk.")

# Load prompt
prompt = hub.pull("rlm/rag-prompt")

# Define state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Generation step
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build LangGraph pipeline
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Ask your question
# response = graph.invoke({"question": "Where can I find the PDF paper of the dasdenoising repository?"})
# response = graph.invoke({"question": "What Repositories can help me with denoising raw waveforms?"})
response = graph.invoke({"question": "Which Repositories can help me with denoising raw waveforms. Please give me at least three."})
print(response["answer"])
