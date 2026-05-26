# 📄 PDF Q&A — RAG System with Web Search 🌐

![Streamlit UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-1C3C3C?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2?style=for-the-badge)
![Tavily](https://img.shields.io/badge/Tavily-Web_Search-4ade80?style=for-the-badge)

A powerful, interactive document Q&A application built using Retrieval-Augmented Generation (RAG). Users can upload PDF documents and ask natural language questions. The system leverages **FAISS** for semantic search over document chunks, **Gemini 2.5 Flash** for grounded answer generation, and optionally supplements the AI's knowledge with live web search results via **Tavily**. 

## ✨ Features

- **Smart PDF Processing:** Upload any PDF, and the app will automatically chunk the text and create sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Semantic Retrieval:** Uses FAISS vector store to retrieve the most relevant document chunks based on user queries.
- **Grounded Generation:** Integrates the Gemini API to generate accurate answers grounded *only* in the provided context, significantly reducing hallucination.
- **Web Search Integration (Tavily):** Option to supplement local PDF knowledge with real-time information from the web.
- **Conversational Memory:** Remembers the last few turns of conversation for follow-up questions and seamless context-building.
- **Polished Dark UI:** A beautifully crafted, responsive Streamlit interface with expandable source citations and document statistics.
- **Optimized Caching:** Built-in resource caching means PDFs are processed once per session, ensuring blazing fast interactions.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/PrathamUdayG/Pdf-RAG-System.git
cd Pdf-RAG-System
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```
> **Note:** The `TAVILY_API_KEY` is optional. If omitted, the web search functionality will be disabled gracefully. You can get a free Tavily API key [here](https://app.tavily.com/).

### 5. Run the Application
```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM / Orchestration:** [LangChain](https://www.langchain.com/) & [Google Gemini](https://deepmind.google/technologies/gemini/)
- **Embeddings:** HuggingFace (`sentence-transformers`)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Web Search:** [Tavily](https://tavily.com/)
- **Language:** Python 3.10+

## 📚 How It Works

1. **Upload & Ingest:** The user uploads a PDF. PyPDFLoader extracts the text, which is then split into overlapping chunks by a `RecursiveCharacterTextSplitter`.
2. **Embed & Index:** Chunks are converted into dense vector embeddings and stored in a local FAISS index.
3. **Retrieve:** Upon a user query, the system retrieves the top *k* most semantically similar chunks from the PDF (and optionally queries Tavily for web results).
4. **Generate:** The Gemini model is prompted with the conversation history, PDF chunks, and web results to synthesize a clear, well-structured answer with verifiable citations.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
