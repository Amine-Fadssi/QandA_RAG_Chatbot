import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# Check if environment variables are set
if not groq_api_key or not hf_token:
    raise ValueError("Please ensure GROQ_API_KEY and HF_TOKEN are set in the .env file.")

# Initialize Model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Set up HuggingFace embeddings
os.environ['HF_TOKEN'] = hf_token
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_website_content(url):
    """Load website content and return documents."""
    try:
        loader = WebBaseLoader(web_paths=(url,))
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"Error loading website content: {e}")
        return []

# Load documents from website (change website from here)
docs = load_website_content("https://www.entrypointai.com/blog/lora-fine-tuning/")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Set up Chroma vector database
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db_")

# Make vectorstore retrievable
retriever = vectorstore.as_retriever()

# Define system prompt template for Q&A
system_prompt = (
    "You are a specialized assistant designed for question-answering tasks. "
    "Leverage the provided context to craft clear, concise, and creatively presented answers. "
    "If the answer is unknown, simply state that you don't have the information. "
    "Limit your response to four sentences for brevity and ensure it is presented in a good format.\n\n{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create QA chain
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# Define system prompt for rephrasing questions with history awareness
history_system_prompt = (
    "You are an assistant responsible for rephrasing questions while maintaining context awareness. "
    "Given the chat history and the latest user question—which may reference previous interactions— "
    "reformulate the question to ensure it stands alone and is fully understandable without needing to refer back to the chat history. "
    "Please refrain from answering the question; focus solely on rephrasing it where necessary or returning it as is if no changes are needed."
)

history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", history_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create history-aware retriever chain
history_aware_retriever = create_history_aware_retriever(llm, retriever, history_prompt)

# Create a retrieval chain that uses history
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Initialize a global dictionary to manage session-based chat histories
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or initialize chat history for a specific session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Function to invoke conversation chain with history
def get_conversational_answer(question: str, session_id: str = "sess_001") -> str:
    """Run the conversational RAG chain with chat history."""
    chat_history = get_session_history(session_id)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    config = {"configurable": {"session_id": session_id}}
    response = conversational_rag_chain.invoke({"input": question}, config=config)
    return response["answer"]


# API request and response models
class QuestionRequest(BaseModel):
    session_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str


# Endpoint to ask a question
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    response = get_conversational_answer(request.question, request.session_id)
    return AnswerResponse(answer=response)


# Run FastAPI app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# Example usage
# input_question = "What is Task Decomposition?"
# answer = get_conversational_answer(input_question)
# print(answer)