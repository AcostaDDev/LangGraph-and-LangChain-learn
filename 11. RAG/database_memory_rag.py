import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import sqlite3
import uuid

from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils.import_loader import import_loader

load_dotenv()
MyPdfLoader = import_loader("pdf_loader")
parser = StrOutputParser()

conn = sqlite3.connect("chat_memory1.db", check_same_thread=False)

cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_history(
        session_id TEXT,
        role TEXT,
        content TEXT
    )
    """
)
conn.commit()

def save_message(session_id: str, role: str, content: str) -> None:
    cursor.execute(
        """
        INSERT INTO chat_history VALUES (?,?,?)
        """,
        (session_id, role, content)
    )
    conn.commit()

def load_chat_history(session_id: str) -> List[BaseMessage]:
    cursor.execute(
        """
        SELECT role, content FROM chat_history WHERE session_id=?
        """,
        (session_id,)
    )

    rows = cursor.fetchall()
    chat_history: List[BaseMessage] = []

    for role, content in rows:
        message_type = HumanMessage if role == "human" else AIMessage
        chat_history.append(message_type(content=content))

    return chat_history
    
def get_all_sessions() -> List[str]:
    cursor.execute(
        """SELECT DISTINCT session_id FROM chat_history ORDER BY rowid DESC"""
    )
    return [row[0] for row in cursor.fetchall()]


# streamlit configs
st.set_page_config(
    page_title="Conversational RAG",
    layout="wide"
)
st.title("Conversarional RAG with memory")

st.sidebar.title("Chats")

if ("session_id" not in st.session_state) or (st.sidebar.button("New Chat")):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

st.sidebar.markdown("Previous Conversations")

for sid in get_all_sessions():
    if st.sidebar.button(sid[:9]):
        st.session_state.session_id = sid
        st.session_state.chat_history = load_chat_history(session_id=sid)

session_id = st.session_state.session_id


# Load and index pdf
@st.cache_resource
def load_vector_store() -> Chroma:
    loader = MyPdfLoader("11. RAG/documents/tfm-david.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(documents=documents)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-miniLM-L6-v2"
    )

    vector_storage = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_storage


vector_store = load_vector_store()
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 7
    }
)

llm = ChatGroq(model="llama-3.3-70b-versatile")

system_message = """
You are a helpful AI assistant. Answer strictly from the provided context.
If the answer is not in present in the context, just say that you don't know.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=system_message
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),
        (
            "human", "Context: {context}.\nQuestion: {input}"
        )
    ],
)

def conversation_rag(user_input: str, chat_history: List[BaseMessage]):
    docs = retriever.invoke(user_input)
    
    context = "\n\n".join(
        f"[Page: {d.metadata.get('page', 'N/A')}] | [PDF TITLE: {d.metadata.get('title', 'No title provided')}]\n{d.page_content}"
        for d in docs
    )

    messages = {
        "input": user_input,
        "context": context,
        "chat_history": chat_history
    }

    chain = prompt | llm | parser

    return chain.invoke(messages), docs

# Load chat history
if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history(session_id)

# Chat window
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("ai").write(msg.content)
    
user_input = st.chat_input("Ask a question from the loaded PDF")

if user_input:
    st.chat_message("user").write(user_input)
    save_message(session_id, "human", user_input)

    ai_answer, sources = conversation_rag(
        user_input=user_input,
        chat_history=st.session_state.chat_history
    )

    st.session_state.chat_history.append(
        HumanMessage(
            content=user_input
        )
    )

    st.chat_message("ai").write(ai_answer)
    save_message(session_id, "ai", ai_answer)

    st.session_state.chat_history.append(
        AIMessage(
            content=ai_answer
        )
    )