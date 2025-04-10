from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os

# Prompt the user for the API key and set it as an environment variable
os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)

# Instead of loading a web page, load the contents of a local text file.
# Replace 'local_page.txt' with the path to your stored file.
local_file_path = "RAG_content/Website/nfdi4ing_html/nfdi4ing.de/home/25_services/25_researchsoftwareengineering/index.html"

try:
    with open(local_file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
except Exception as e:
    print(f"Error reading {local_file_path}: {str(e)}")
    file_content = ""

# Wrap the text in a Document with metadata (optional)
doc = Document(page_content=file_content, metadata={"source": local_file_path})
docs = [doc]

# Chunk the document content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index the chunks in the vector store
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Preview
response = graph.invoke({"question": "how can you help me to find research software?"})
print(response["answer"])
