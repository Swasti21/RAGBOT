
📚 Multi-PDF Chat Agent 🤖

Meet Multi-PDF Chat AI App! 🚀 Chat seamlessly with multiple PDFs using LangChain, Google Gemini, HuggingFace embeddings & FAISS Vector DB, all wrapped in a clean Streamlit UI.

Get fast, accurate, natural answers from your documents — ask questions in plain language, and the AI finds + explains the answer from your PDFs. 🔥✨

⸻

📝 Description

The Multi-PDF Chat Agent is a Streamlit-based web application that allows you to:
	•	Upload multiple PDF documents.
	•	Automatically extract and chunk text.
	•	Build a searchable FAISS vector index.
	•	Ask questions in real-time and get evidence-based answers.

Think of it as ChatGPT for your documents.

⸻

📊 Demo (Optional for Now)

⚡ Soon you’ll be able to deploy this with Streamlit Cloud or Hugging Face Spaces.

⸻

🎯 How It Works

	1.	PDF Loading → Extracts raw text from your uploaded PDFs.
	2.	Text Chunking → Splits into small sections so AI can handle them efficiently.
	3.	Embeddings → Uses HuggingFace embeddings (all-MiniLM-L6-v2) to vectorize text.
	4.	Vector DB (FAISS) → Stores embeddings for fast similarity search.
	5.	LLM (Gemini) → Generates answers by combining retrieved evidence with generative reasoning.

⸻

🚀 Features
	•	✅ Multi-PDF conversational QA
	•	✅ HuggingFace embeddings (free, unlimited, no API quota issues)
	•	✅ Google Gemini LLM (powerful, contextual answers)
	•	✅ Natural, ChatGPT-like responses (not JSON blobs)
	•	✅ Evidence view → See where the answer came from
	•	✅ Streamlit UI with polished design

⸻

⚙️ Requirements
	•	streamlit – interactive web UI
	•	google-generativeai – Gemini integration
	•	python-dotenv – load environment variables
	•	langchain + langchain-community – RAG pipelines
	•	PyPDF2 – PDF text extraction
	•	faiss-cpu – vector database
	•	sentence-transformers – HuggingFace embeddings
	•	langchain-google-genai – Gemini LLM wrapper

⸻

▶️ Installation

Clone the repo:

git clone https://github.com/Adityism/Ragbot.git
cd Ragbot

Install dependencies:

pip install -r requirements.txt

Set up your .env with your Gemini API key:

GOOGLE_API_KEY=<your-api-key>

Run the app:

streamlit run chatapp.py


⸻

💡 Usage
	1.	Upload one or more PDFs in the sidebar.
	2.	Click Build Index to process them.
	3.	Ask any question in plain English.
	4.	Get answers + evidence directly from your docs.

⸻

📁 Project Structure

Ragbot/
├── RAG-BOT/               # Core app directory
│   ├── docs/              # Your uploaded PDFs
│   ├── faiss_index/       # Vector DB storage
│   └── chatapp.py         # Main Streamlit app
├── requirements.txt
├── .env                   # Your API key
└── README.md


⸻

🪪 License

Distributed under the MIT License. See LICENSE for details.

⸻

⭐ Support

If you like this project:
	•	Drop a ⭐ on the repo → Adityism/Ragbot
	•	Connect with me:
	•	LinkedIn
	•	GitHub

