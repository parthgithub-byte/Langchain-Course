# 🦜🔗 LangChain & Modern LLM Workflows

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white" alt="Gemini" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
  <img src="https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=python&logoColor=white" alt="uv" />
</div>

<br />

A repository dedicated to exploring AI application development and hybrid LLM workflows utilizing **Google Gemini** (for high-performance cloud inference) and **Ollama** (for secure, local open-weights models) powered by the **LangChain** framework.

⚡ This project is optimized using [uv](https://github.com/astral-sh/uv) for lightning-fast Python dependency and environment management.

---

## 📖 About This Repository

This project serves as a comprehensive log of my learning journey through LangChain. It focuses on the architectural differences and practical applications of alternating between cloud-based commercial models and local, private models.

### 🎯 Core Concepts Explored
* **Hybrid Model Integration:** Routing tasks between Google's Gemini API and local open-source models (like LLaMA 3, Mistral) running via Ollama.
* **Retrieval Augmented Generation (RAG):** "Chatting" with custom documents using optimized document loaders, text splitters, and embedding models.
* **Vector Databases:** Storing and querying high-dimensional semantic data using local vector stores like ChromaDB or FAISS.
* **Conversational Memory:** Implementing short-term and long-term memory structures to maintain context across multiple turns.
* **Agents & Tool Calling:** Building AI agents capable of dynamically routing tasks and using external APIs to solve complex queries.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Package Manager:** `uv` (by Astral)
* **Framework:** LangChain
* **Cloud LLM:** Google Gemini API
* **Local LLM:** Ollama (LLaMA 3, Mistral, etc.)
* **Environment:** `.env` for secure credential management

---

## 🚀 Getting Started (Local Setup)

This project uses `uv` to make environment setup incredibly fast. 

**1. Clone the repository:**
```bash
git clone [https://github.com/parthgithub-byte/Langchain-Course.git](https://github.com/parthgithub-byte/Langchain-Course.git)
cd Langchain-Course
```

**2. Setup Environment with `uv`:
If you don't have `uv` installed, [install it here](https://github.com/astral-sh/uv).  
```bash
# Create a virtual environment instantly
uv venv

# Activate it (Mac/Linux)
source .venv/bin/activate
# Or on Windows: .venv\Scripts\activate

# Install dependencies blazingly fast
uv pip install -r requirements.txt
```

**3. Set up Ollama (For Local Models):**
Ensure you have [Ollama](https://ollama.com/) installed and running on your machine. Pull the model you want to experiment with:
```bash
ollama run llama3
```

**4. Secure your API Keys:**
Create a `.env` file in the root directory. Important: Ensure `.env` is listed in your `.gitignore` file so you do not accidentally push your keys to GitHub!
```
# Google Gemini API Key
GEMINI_API_KEY="AIzaSyYourGeminiKeyHere..."

# LangSmith (Optional: for tracing and debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="ls__your_langchain_key..."
```

## 👨‍💻 Author

**Parth Pakhare**
* Computer Engineering Student | Full-Stack & AI Enthusiast
* GitHub: [@parthgithub-byte](https://github.com/parthgithub-byte)
