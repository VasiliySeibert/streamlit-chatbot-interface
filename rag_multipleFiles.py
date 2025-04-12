import os
import glob
import nltk
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.document_loaders import UnstructuredHTMLLoader
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

# Set up persistent NLTK data directory and download required packages
os.environ["NLTK_DATA"] = "/tmp/nltk_data"
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Ensure that the API key is available.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")


llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# Define the directory for storing the FAISS index.
FAISS_INDEX_DIR = "faiss_index"

# Check if a persistent FAISS index exists, then load or create a new one.
if os.path.exists(FAISS_INDEX_DIR):
    print("Loading FAISS vector store from disk...")
    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS vector store...")
    # Locate HTML files in the target directory.
    directory_path = "RAG_content/Website/nfdi4ing_html"
    html_files = glob.glob(os.path.join(directory_path, "**/*.html"), recursive=True)

    # Filter out files that are too large if needed (max 2MB here).
    def should_process(file_path, max_size=2 * 1024 * 1024):
        try:
            return os.path.getsize(file_path) < max_size
        except Exception:
            return False

    filtered_files = [f for f in html_files if should_process(f)]
    docs = []

    # Load documents from the HTML files.
    for file in filtered_files:
        try:
            loader = UnstructuredHTMLLoader(file, mode="elements")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not docs:
        print("No documents loaded.")
        exit(1)
    else:
        # Split documents into chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        # Create a FAISS index from the split documents.
        vector_store = FAISS.from_documents(all_splits, embeddings)
        # Persist the FAISS index to disk.
        vector_store.save_local(FAISS_INDEX_DIR)
        print("FAISS vector store saved to disk.")

# Pull the retrieval-augmented generation (RAG) prompt.
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    # Use the vector store to perform similarity search.
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    # Concatenate document content and invoke the prompt and LLM.
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build and compile the state graph.
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Invoke the graph with a sample question.
response = graph.invoke({"question": "when did nfdi4ing start?"})
print(response["answer"])
