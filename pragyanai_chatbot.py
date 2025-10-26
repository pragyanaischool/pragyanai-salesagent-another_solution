# pragyanai_chatbot.py
import os
from datetime import datetime
from typing import Dict, Optional
import streamlit as st
from langchain_groq import ChatGroq
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
import re

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = "pragyanai_sales"
    COLLECTION_NAME = "student_leads"
    CHAT_HISTORY_COLLECTION = "chat_histories"
    PDF_DIRECTORY = "./program_docs"
    MODEL_NAME = "llama-3.1-70b-versatile"

# MongoDB Manager
class MongoDBManager:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.leads_collection = self.db[Config.COLLECTION_NAME]

    def save_student_lead(self, student_data: Dict) -> str:
        student_data['created_at'] = datetime.utcnow()
        student_data['updated_at'] = datetime.utcnow()
        result = self.leads_collection.insert_one(student_data)
        return str(result.inserted_id)

# RAG System for PDF Knowledge Base
class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.retriever = None

    def load_and_process_pdfs(self, pdf_directory: str):
        if os.path.exists(pdf_directory):
            loader = DirectoryLoader(
                pdf_directory,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

    def get_relevant_context(self, query: str) -> str:
        if self.retriever is None:
            return ""
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])

# PragyanAI Agent Logic
class PragyanAIAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=0.7
        )
        self.rag_system = RAGSystem()
        self.db_manager = MongoDBManager()
        self.required_fields = [
            "name", "email", "phone", "college",
            "branch", "semester", "academic_score"
        ]

    def initialize_rag(self):
        self.rag_system.load_and_process_pdfs(Config.PDF_DIRECTORY)

    def get_system_prompt(self, student_info: Dict, context: str = "") -> str:
        base_prompt = f"""You are an enthusiastic AI admissions assistant for PragyanAI.
Collect student data conversationally, explain AI value, present programs, and recommend based on profile.

Student info:
{self._format_student_info(student_info)}

Knowledge base context:
{context}

Be friendly and ask one question at a time. Output personalized recommendation after all info collected."""
        return base_prompt

    def _format_student_info(self, info: Dict) -> str:
        return "\n".join(
            f"- {field.replace('_', ' ').title()}: {info.get(field, 'Not collected')}"
            for field in self.required_fields
        )

    def extract_information(self, message: str, current_info: Dict) -> Dict:
        updates = {}
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+91|91)?[\s-]?[6-9]\d{9}\b'
        score_pattern = r'\b(?:cgpa|gpa|score|percentage)[\s:]*(\d+\.?\d*)\b'

        if not current_info.get('email'):
            email_match = re.search(email_pattern, message)
            if email_match:
                updates['email'] = email_match.group()

        if not current_info.get('phone'):
            phone_match = re.search(phone_pattern, message)
            if phone_match:
                updates['phone'] = phone_match.group()

        if not current_info.get('academic_score'):
            score_match = re.search(score_pattern, message.lower())
            if score_match:
                updates['academic_score'] = score_match.group(1)

        # Generic fallback to capture name, college, branch, semester if message short and no info yet
        for field in ['name', 'college', 'branch', 'semester']:
            if not current_info.get(field) and field in message.lower():
                updates[field] = message.strip()

        return updates

    def determine_next_question(self, student_info: Dict) -> Optional[str]:
        for field in self.required_fields:
            if not student_info.get(field):
                return field
        return None

    def generate_program_recommendation(self, student_info: Dict) -> str:
        context = self.rag_system.get_relevant_context(
            f"Program recommendation for {student_info.get('branch', 'engineering')} student in semester {student_info.get('semester', 'N/A')}"
        )
        prompt = f"""Recommend the BEST PragyanAI program based on:

Profile:
- Branch: {student_info.get('branch', 'N/A')}
- Semester: {student_info.get('semester', 'N/A')}
- Academic Score: {student_info.get('academic_score', 'N/A')}

Programs:
1. Generative AI Bootcamp
2. Agentic AI Workshop
3. Machine Learning Foundations
4. Deep Learning Specialization
5. End-to-End MLOps

Knowledge base context:
{context}

Provide a clear recommendation in 2-3 sentences."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

# Streamlit App
def main():
    st.set_page_config(page_title="PragyanAI Sales Chatbot", layout="wide")
    st.title("🤖 PragyanAI Agentic Sales Chatbot")
    st.image("PragyanAI_Transperent.png")
    st.markdown("*AI-Powered Student Engagement System*")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().timestamp()}"
    if 'student_info' not in st.session_state:
        st.session_state.student_info = {}
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        agent = PragyanAIAgent()
        agent.initialize_rag()
        st.session_state.agent = agent

    # MongoDB chat history
    chat_history = MongoDBChatMessageHistory(
        session_id=st.session_state.session_id,
        connection_string=Config.MONGODB_URI,
        database_name=Config.DATABASE_NAME,
        collection_name=Config.CHAT_HISTORY_COLLECTION
    )

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📋 Student Information")
        for field in st.session_state.agent.required_fields:
            value = st.session_state.student_info.get(field, "-")
            st.text(f"{field.replace('_', ' ').title()}: {value}")

        if len(st.session_state.student_info) == len(st.session_state.agent.required_fields):
            st.success("✅ All information collected!")
            if st.button("Generate Recommendation"):
                rec = st.session_state.agent.generate_program_recommendation(st.session_state.student_info)
                st.info(rec)
                st.session_state.student_info['recommendation'] = rec
                st.session_state.agent.db_manager.save_student_lead(
                    {**st.session_state.student_info, "session_id": st.session_state.session_id}
                )

    with col1:
        st.subheader("💬 Chat")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if len(st.session_state.messages) == 0:
            greeting = "Hello! 👋 I'm your PragyanAI admissions assistant. May I have your name?"
            st.session_state.messages.append({"role": "assistant", "content": greeting})
            chat_history.add_ai_message(greeting)
            st.experimental_rerun()

        if user_input := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            chat_history.add_user_message(user_input)

            extracted = st.session_state.agent.extract_information(user_input, st.session_state.student_info)
            st.session_state.student_info.update(extracted)

            next_question = st.session_state.agent.determine_next_question(st.session_state.student_info)
            if next_question:
                follow_up = {
                    "name": "May I have your full name?",
                    "email": "Please provide your email address.",
                    "phone": "Could you share your phone number?",
                    "college": "Which college are you studying at?",
                    "branch": "What's your branch or major?",
                    "semester": "Which semester are you currently in?",
                    "academic_score": "What's your current CGPA or academic score?"
                }[next_question]
            else:
                follow_up = "All information received. You can generate your program recommendation on the right."

            context = st.session_state.agent.rag_system.get_relevant_context(user_input)
            prompt = st.session_state.agent.get_system_prompt(st.session_state.student_info, context)

            response = st.session_state.agent.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=user_input)
            ])

            st.session_state.messages.append({"role": "assistant", "content": response.content + "\n\n" + follow_up})
            chat_history.add_ai_message(response.content + "\n\n" + follow_up)

            st.experimental_rerun()

if __name__ == "__main__":
    main()
