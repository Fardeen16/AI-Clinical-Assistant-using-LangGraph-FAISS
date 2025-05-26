# AI-Clinical-Assistant-using-LangGraph-FAISS

## 🩺 AI-powered Clinical Intake Assistant + Diagnostic Triage System
A LangGraph-based conversational medical assistant that leverages Retrieval-Augmented Generation (RAG), FAISS vector search, and large language models (LLMs) to streamline patient intake, triage symptoms, and visually interpret uploaded medical reports — all through an intuitive Streamlit interface.

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

📁 Project Structure

`Medic_AI_Chatbot/`
`├── connect_memory_with_llm.py   # Phase 2 - LangGraph + FAISS + Mistral
├── embed_pdfs_to_faiss.py       # Phase 1 - PDF loader + chunking + FAISS
├── medicbot_app.py              # Phase 3 - Streamlit chatbot UI
├── vectorstore/
│   └── db_faiss/                # Stored vector embeddings (FAISS index)
├── data/                        # Medical reference PDFs
├── requirements.txt
└── README.md`
<br/>


## ⚙️ Setup Instructions
1. Clone the repo
git clone https://github.com/yourusername/medic-ai-chatbot.git
cd medic-ai-chatbot
2. Create and activate virtual environment
bash
Copy
Edit
python -m venv medicEnv
source medicEnv/bin/activate   # or .\medicEnv\Scripts\activate on Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set HuggingFace Token
Create a .env file or export your token:

env
Copy
Edit
HF_TOKEN=your_huggingface_token_here
▶️ Run the App
Embed medical PDFs (only once):
bash
Copy
Edit
python embed_pdfs_to_faiss.py
Start chatbot:
bash
Copy
Edit
streamlit run medicbot_app.py
📸 Screenshots
Chatbot Interface	Report Visualization

🧑‍⚕️ Use Cases
Clinical triage in telemedicine apps

Medical chatbots for hospitals or diagnostics

AI-assisted patient data intake

Medical education or decision support tools

🧩 Future Enhancements
🔍 ICD-10 or SNOMED disease code suggestions

🗂️ Integration with electronic health records (EHR)

🧾 Summarization of uploaded lab reports

🌐 Multilingual symptom triage

📜 License
This project is licensed under the MIT License.

🙌 Acknowledgements
LangGraph

HuggingFace

FAISS by Facebook AI

Streamlit

Plotly
