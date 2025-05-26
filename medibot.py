import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from connect_memory_with_llm import compiled_graph  # 👈 Import your LangGraph agent
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def main():
    st.set_page_config(page_title="MedicBot", page_icon="🧬")
    st.title("🩺 MedicBot – Ask Your Medical Assistant")

    # Initialize session message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Describe your symptoms or ask a medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Send the prompt through LangGraph agent
            graph_input = {"messages": [HumanMessage(content=prompt)]}
            result = compiled_graph.invoke(graph_input)
            answer_msg = result["messages"][-1]

            # Show response in UI
            st.chat_message("assistant").markdown(answer_msg.content)
            st.session_state.messages.append({'role': 'assistant', 'content': answer_msg.content})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
