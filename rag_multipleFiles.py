import os
import glob
import pickle
import nltk
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.document_loaders import UnstructuredHTMLLoader

# Set up persistent NLTK data directory and download required packages
os.environ["NLTK_DATA"] = "/tmp/nltk_data"
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

VECTOR_STORE_FILE = "vector_store.pkl"

if os.path.exists(VECTOR_STORE_FILE):
    print("Loading vector store from disk...")
    with open(VECTOR_STORE_FILE, "rb") as f:
        vector_store = pickle.load(f)
else:
    print("Creating new vector store...")
    vector_store = InMemoryVectorStore(embeddings)

    directory_path = "RAG_content/Website/nfdi4ing_html"
    html_files = glob.glob(os.path.join(directory_path, "**/*.html"), recursive=True)

    # Filter out files that are too large if needed
    def should_process(file_path, max_size=2 * 1024 * 1024):  # max 2MB
        try:
            return os.path.getsize(file_path) < max_size
        except Exception:
            return False

    filtered_files = [f for f in html_files if should_process(f)]
    docs = []

    for file in filtered_files:
        try:
            # Use a possibly faster mode if available; otherwise, default to the provided mode.
            loader = UnstructuredHTMLLoader(file, mode="elements")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not docs:
        print("No documents loaded.")
        exit(1)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        _ = vector_store.add_documents(documents=all_splits)

        with open(VECTOR_STORE_FILE, "wb") as f:
            pickle.dump(vector_store, f)
        print("Vector store saved to disk.")

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "what is interesting about nfdi4ing_html content?"})
print(response["answer"])
