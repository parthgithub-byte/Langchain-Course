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
* **Hybrid Model Integration:** Routing tasks between Google's Gemini API, local open-weights models via Ollama, and cloud models via OpenRouter.
* **Agents, from the Outside In:** The same ReAct (reason → act → observe) agent taken apart across three layers — Layer 0 uses LangChain's prebuilt `create_agent` (batteries included, web search via Tavily, structured output via Pydantic); Layer 1 rebuilds that same loop by hand with LangChain (`bind_tools`, `@tool`); Layer 2 strips LangChain out entirely and rebuilds it again on the raw Ollama SDK, at the protocol level.
* **Tracing & Observability:** Instrumenting both LangChain and raw-SDK code paths with LangSmith (`@traceable`) to compare automatic vs. manual tracing.

### 🗺️ Planned / Not Yet Built

* **Retrieval Augmented Generation (RAG)** and **Vector Databases** (ChromaDB/FAISS) — chatting with custom documents.
* **Conversational Memory** — short-term and long-term context across turns.

## 📚 Notes Index

Each stage of the course has a companion `.md` writeup living next to its code, explaining the concepts and design decisions in depth. This table is the map — update it whenever a new stage's notes file is added.

**The layer numbers trace an abstraction arc, not a feature-count sequence:** Layer 0 is the most abstracted (prebuilt agent), Layer 1 rebuilds it by hand within LangChain, and Layer 2 strips LangChain out entirely. Read 0 → 1 → 2 to see the same agent from the outside in.

| Stage | Topic | Code | Notes |
|---|---|---|---|
| Basics | Prompt → Model → Chain | [`main1.py`](main1.py), [`main2.py`](main2.py) | [main1_main2_langchain_basics.md](main1_main2_langchain_basics.md) |
| Layer 0 | Prebuilt Agents (`create_agent`) + Web Search + Structured Output | [`Layer 0 SearchAgent/`](Layer%200%20SearchAgent/) | [layer0search_agents_tavily_structured_output.md](Layer%200%20SearchAgent/layer0search_agents_tavily_structured_output.md) |
| Layer 1 | ReAct Agent Loop, Hand-Built (LangChain) | [`Layer 1 E-Commerce Agent/`](Layer%201%20E-Commerce%20Agent/) | [layer1agent_langchain_react_loop.md](Layer%201%20E-Commerce%20Agent/layer1agent_langchain_react_loop.md) |
| Layer 2 | Same Agent, No LangChain (Raw Ollama SDK) | [`Layer 2 E-Commerce Agent/`](Layer%202%20E-Commerce%20Agent/) | [layer2agent_raw_ollama_sdk.md](Layer%202%20E-Commerce%20Agent/layer2agent_raw_ollama_sdk.md) |
| Layer 3 | *(upcoming)* | — | — |

**Convention for new stages:** name the notes file `layerN<topic>.md`, keep it in the same folder as the code it documents, and add a row here.

---

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **Package Manager:** `uv` (by Astral)
* **Framework:** LangChain — with Layer 2 deliberately dropping it for the raw Ollama SDK, to show what the framework abstracts away
* **Cloud LLM:** Google Gemini API, OpenRouter
* **Local LLM:** Ollama (e.g. `qwen3:1.7b`)
* **Search Tooling:** Tavily (`langchain-tavily`, `tavily-python`)
* **Tracing:** LangSmith (`@traceable`)
* **Environment:** `.env` for secure credential management

---

## 🚀 Getting Started (Local Setup)

This project uses `uv` to make environment setup incredibly fast. 

**1. Clone the repository:**
```bash
git clone https://github.com/parthgithub-byte/Langchain-Course.git
cd Langchain-Course
```

**2. Setup Environment with `uv`:**
If you don't have `uv` installed, [install it here](https://github.com/astral-sh/uv).  
```bash
# Installs Python 3.13, creates .venv, and installs all locked dependencies
uv sync

# Activate it (Mac/Linux)
source .venv/bin/activate
# Or on Windows:
.venv\Scripts\activate
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
GOOGLE_API_KEY="AIzaSyYourGeminiKeyHere..."

# Tavily (required for Layer 0's web search tooling)
TAVILY_API_KEY="tvly-your_tavily_key..."

# LangSmith (Optional: for tracing and debugging)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="lsv2_your_langsmith_key..."
LANGSMITH_PROJECT="your-project-name"
```

## 👨‍💻 Author

**Parth Pakhare**
* Computer Engineering Student | Full-Stack & AI Enthusiast
* GitHub: [@parthgithub-byte](https://github.com/parthgithub-byte)
