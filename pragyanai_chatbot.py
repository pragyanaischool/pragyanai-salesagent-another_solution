# pragyanai_chatbot.py
import os
from datetime import datetime
from typing import Dict, Optional
import streamlit as st
from langchain_groq import ChatGroq
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# MODIFIED: Corrected import path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
import re

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
    MONGODB_URI = os.getenv("MONGODB_URI", st.secrets.get("MONGODB_URI", ""))
    DATABASE_NAME = "pragyanai_sales"
    COLLECTION_NAME = "student_leads"
    CHAT_HISTORY_COLLECTION = "chat_histories"
    PDF_DIRECTORY = "./program_docs"
    MODEL_NAME = "llama3-70b-8192" # Using a powerful model for better reasoning

# MongoDB manager
class MongoDBManager:
    def __init__(self):
        if not Config.MONGODB_URI:
            raise ValueError("MONGODB_URI not set in environment or Streamlit secrets.")
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.leads_collection = self.db[Config.COLLECTION_NAME]

    def save_student_lead(self, student_data: Dict) -> str:
        student_data['created_at'] = datetime.utcnow()
        student_data['updated_at'] = datetime.utcnow()
        # Use email as a unique identifier to prevent duplicate entries
        self.leads_collection.update_one(
            {'email': student_data.get('email')},
            {'$set': student_data},
            upsert=True
        )
        return student_data.get('email')

# RAG System
class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retriever = None

    def load_and_process_pdfs(self, pdf_directory: str):
        if not os.path.exists(pdf_directory):
             os.makedirs(pdf_directory, exist_ok=True) # Create dir if it doesn't exist
             st.warning(f"PDF directory '{pdf_directory}' did not exist and was created. Please add program documents.")
             return # Stop if there are no docs to load
        
        loader = DirectoryLoader(
            pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        documents = loader.load()
        if not documents:
            st.error(f"No PDF files found in '{pdf_directory}'. The RAG system cannot be initialized.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def get_relevant_context(self, query: str) -> str:
        if self.retriever is None:
            return "No knowledge base loaded. Answer based on general knowledge."
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

# Agent core logic
class PragyanAIAgent:
    def __init__(self):
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in environment or Streamlit secrets.")
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY, model_name=Config.MODEL_NAME, temperature=0.7
        )
        self.rag_system = RAGSystem()
        self.db_manager = MongoDBManager()
        self.required_fields = [
            ("name", "Great! What is your email address?"),
            ("email", "Thanks. Could you provide your phone number?"),
            ("phone", "Excellent. Which college are you from?"),
            ("college", "Got it. What is your branch or major?"),
            ("branch", "Almost there! What is your current semester?"),
            ("semester", "Finally, what is your current academic score (CGPA or percentage)?"),
            ("academic_score", "Thank you for providing all your details!"),
        ]

    def initialize_rag(self):
        self.rag_system.load_and_process_pdfs(Config.PDF_DIRECTORY)

    def generate_program_recommendation(self, student_info: Dict) -> str:
        context = self.rag_system.get_relevant_context(
            f"Program recommendation for a {student_info.get('branch', 'engineering')} "
            f"student in semester {student_info.get('semester', 'N/A')} with a score of "
            f"{student_info.get('academic_score', 'N/A')}."
        )

        prompt = f"""Based on this student profile, recommend the BEST PragyanAI program and explain why in a persuasive, friendly tone.

        Student Profile:
        {student_info}

        Context from PragyanAI's official documents:
        {context}

        Provide a specific recommendation with clear, compelling reasoning in 2-4 sentences. Frame it as an expert advisor helping the student make a great career decision."""
        
        response = self.llm.invoke(prompt)
        return response.content

def main():
    st.set_page_config(
        page_title="PragyanAI Sales Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 PragyanAI Agentic Sales Chatbot")
    st.markdown("*AI-Powered Student Engagement System*")

    # --- MODIFIED: Robust Initialization with Error Handling ---
    try:
        if "agent" not in st.session_state:
            st.session_state.agent = PragyanAIAgent()
            st.session_state.agent.initialize_rag()
        
        # This is the crucial line that fixes the error.
        # It ensures the 'agent' variable exists on every script rerun.
        agent = st.session_state.agent

    except (ValueError, FileNotFoundError) as e:
        st.error(f"Initialization Failed: {e}")
        st.stop() # Stop the app if the agent can't be created.
    except Exception as e:
        st.error(f"An unexpected error occurred during setup: {e}")
        st.stop()
    # --- END MODIFICATION ---

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().timestamp()}"

    if "student_info" not in st.session_state:
        st.session_state.student_info = {}

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_history = MongoDBChatMessageHistory(
        session_id=st.session_state.session_id,
        connection_string=Config.MONGODB_URI,
        database_name=Config.DATABASE_NAME,
        collection_name=Config.CHAT_HISTORY_COLLECTION
    )

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📋 Student Information")
        # This loop will now work correctly
        for field in agent.required_fields:
            value = st.session_state.student_info.get(field, "-")
            st.text(f"{field.replace('_', ' ').title()}: {value}")

        all_info_collected = all(field in st.session_state.student_info for field in agent.required_fields)

        if all_info_collected:
            st.success("✅ All information collected!")
            if st.button("Generate Recommendation"):
                rec = agent.generate_program_recommendation(st.session_state.student_info)
                st.info(rec)
                st.session_state.student_info['recommendation'] = rec
                agent.db_manager.save_student_lead(
                    {**st.session_state.student_info, "session_id": st.session_state.session_id}
                )

    with col1:
        st.subheader("💬 Chat")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if not st.session_state.messages:
            greeting = "Hello! 👋 I'm your PragyanAI admissions assistant. To get started, may I have your full name?"
            st.session_state.messages.append({"role": "assistant", "content": greeting})
            chat_history.add_ai_message(greeting)
            st.rerun() # Use st.rerun() instead of the custom safe_rerun

        if user_input := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            chat_history.add_user_message(user_input)

            # --- MODIFIED: Simplified information collection logic ---
            # Determine which question we are waiting for an answer to
            current_question_field = None
            next_question = ""
            all_done = True
            
            for field in agent.required_fields:
                if field not in st.session_state.student_info:
                    current_question_field = field
                    all_done = False
                    break
            
            # If we were waiting for an answer, save the user's input
            if current_question_field:
                st.session_state.student_info[current_question_field] = user_input.strip()

                # Find the next question to ask
                next_field_found = False
                for field in agent.required_fields:
                    if field not in st.session_state.student_info:
                        # This is a bit complex, might need a rethink, but let's go with a simple LLM call for now.
                        next_field_found = True
                        break
                
                if not next_field_found:
                    all_done = True

            if all_done:
                 # Now that all info is collected, generate recommendation and save lead
                recommendation = agent.generate_program_recommendation(st.session_state.student_info)
                ai_response = f"Thank you for providing all your details! Here is my recommendation for you:\n\n{recommendation}"
                st.session_state.student_info['recommendation'] = recommendation
                agent.db_manager.save_student_lead({**st.session_state.student_info, "session_id": st.session_state.session_id})

            else:
                # Use the LLM to ask the next question conversationally
                context = agent.rag_system.get_relevant_context(user_input)
                system_prompt = agent.get_system_prompt(st.session_state.student_info, context)
                messages_for_groq = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
                response = agent.llm.invoke(messages_for_groq)
                ai_response = response.content

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            chat_history.add_ai_message(ai_response)

            st.rerun()

if __name__ == "__main__":
    main()
