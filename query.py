from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

# 1. Load the saved vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

# 2. Set up the retriever (fetch top 3 most relevant chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Set up the LLM
# Load API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it before running.")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 4. Build the RAG chain
system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Ask questions in a loop
print("RAG system ready. Type 'quit' to exit.\n")
while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break

    # Get answer from chain
    answer = qa_chain.invoke(question)
    print(f"\nAnswer: {answer}")
    
    # Get source documents separately
    docs = retriever.invoke(question)
    print(f"\nSources used:")
    for doc in docs:
        src = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', '')
        print(f"  - {src} (page {page})")
    print()