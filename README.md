# AI-Clinical-Assistant-using-LangGraph-FAISS
<br/>

## 🩺 AI-powered Clinical Intake Assistant + Diagnostic Triage System
A LangGraph-based conversational medical assistant that leverages Retrieval-Augmented Generation (RAG), FAISS vector search, and large language models (LLMs) to streamline patient intake, triage symptoms, and visually interpret uploaded medical reports — all through an intuitive Streamlit interface.
<br/>


## 🔍 Overview
This project simulates an intelligent clinical assistant capable of:
- Conversationally collecting patient symptoms and history
- Using RAG with FAISS to retrieve relevant medical information
- Responding with LLM-powered diagnosis suggestions
- Visually plotting uploaded medical reports using Plotly for clear insights
<br/>

## 🧠 Tech Stack & Tools
| Component |	Technology Used |
| --- | --- |
| Language Model | 🤖 [Mistral-7B-Instruct (HuggingFace)] |
| Retrieval Engine |	🧠 FAISS + HuggingFace Embeddings |
| Orchestration |	🔁 LangGraph (LangChain agents) |
| Frontend UI |	🖥️ Streamlit |
| Visualization |	📊 Plotly (for graphical report insights) |
| Data Preprocessing |	📚 PDF Loaders + Chunking |
<br/>

## 🚀 Features
- ✅ LangGraph-powered memory graph for intelligent multi-turn conversations
- ✅ RAG-based symptom triage from medical PDFs using vector retrieval
- ✅ LLM integration via HuggingFace InferenceEndpoint (Mistral-7B)
- ✅ Streamlit chatbot UI with persistent session history
- ✅ Medical report visualization using Plotly for enhanced interpretation
<br/>

## 📁 Project Structure

```
Medic_AI_Chatbot/
├── connect_memory_with_llm.py   # Phase 2 - LangGraph + FAISS + Mistral
├── embed_pdfs_to_faiss.py       # Phase 1 - PDF loader + chunking + FAISS
├── medicbot_app.py              # Phase 3 - Streamlit chatbot UI
├── vectorstore/
│   └── db_faiss/                # Stored vector embeddings (FAISS index)
├── data/                        # Medical reference PDFs
├── requirements.txt
└── README.md
```
<br/>

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/5488619f-d795-4507-9ba7-ceaaa3620f5f)
<br/>

## 🧑‍⚕️ Use Cases
- Clinical triage in telemedicine apps
- Medical chatbots for hospitals or diagnostics
- AI-assisted patient data intake
- Medical education or decision support tools
