from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tools = [search]
agent = create_agent(model=llm,tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Pune or Mumbai area on linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()

# Remember to use the model gemini-3-flash-preview here. It is slower, but handles agentic workflow. 
# Output:
# Based on current LinkedIn listings, here are three job postings for AI Engineers specializing in LangChain in the Pune and Mumbai areas:

# ### 1. AI Engineer – LLMs, LangChain, Agentic Systems
# *   **Company:** Shivsys Incorporation
# *   **Location:** Hybrid – Pune / Mumbai (also Bengaluru, Chennai, Noida)
# *   **Experience Level:** 4–6 Years
# *   **Key Skills:** LLMs, LangChain, Agentic Systems, RAG (Retrieval-Augmented Generation).
# *   **Job Description:** This role focuses on building sophisticated agentic systems and applications powered by Large Language Models. The candidate is expected to have deep proficiency in using LangChain for orchestrating complex AI workflows and integrating them into enterprise-level solutions.
# *   **Link:** [View on LinkedIn](https://in.linkedin.com/jobs/view/ai-engineer-%E2%80%93-llms-langchain-agentic-systems-at-shivsys-incorporation-4212241326)

# ### 2. Artificial Intelligence Engineer
# *   **Company:** AntiChurn
# *   **Location:** Pune, Maharashtra (Remote/On-site details vary)
# *   **Key Skills:** Python, Prompt Engineering, LangChain, GPT models.
# *   **Job Description:** AntiChurn is looking for a skilled AI engineer to develop and optimize AI-driven customer retention tools. The role involves heavy use of LangChain for application development, alongside a strong focus on prompt engineering and fine-tuning model outputs for specific business contexts.
# *   **Link:** [View on LinkedIn](https://in.linkedin.com/jobs/view/artificial-intelligence-engineer-at-antichurn-4265951512)

# ### 3. AI Engineer - GPT / LangChain / RAG / Data Pipelines
# *   **Company:** Peak Trust Global Real Estate
# *   **Location:** Pune, Maharashtra
# *   **Key Skills:** GPT, LangChain, RAG, Data Pipelines, Vector Databases.
# *   **Job Description:** This position is geared toward building AI solutions for the real estate sector. The focus is on creating RAG (Retrieval-Augmented Generation) pipelines using LangChain to query large datasets of property information and legal documents, ensuring high accuracy and performance in data retrieval.
# *   **Link:** [View on LinkedIn](https://in.linkedin.com/jobs/view/ai-engineer-gpt-langchain-rag-data-pipelines-at-peak-trust-global-real-estate-4302171754)

# ---
# **Tips for your search:**
# *   **Keyword Variations:** If you are searching for more, try using terms like "Generative AI Engineer" or "LLM Developer" as many LangChain-heavy roles are categorized under these titles.
# *   **Direct Search:** You can refine these further on LinkedIn by filtering for "Past Week" to find the most active recruiters.