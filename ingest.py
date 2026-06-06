from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, glob

load_dotenv()

# 1. Load documents from /docs folder
docs = []
for path in glob.glob("docs/*.pdf"):
    docs.extend(PyPDFLoader(path).load())
for path in glob.glob("docs/*.txt"):
    docs.extend(TextLoader(path).load())

print(f"Loaded {len(docs)} document pages")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50      # overlap to avoid losing context at boundaries
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# 3. Embed using a free local Hugging Face model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in FAISS and save to disk
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Vector store saved to faiss_index/")