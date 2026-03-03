from dotenv import load_dotenv
import os
load_dotenv()

# This ignores whatever LANGSMITH_PROJECT was in the .env
os.environ["LANGSMITH_PROJECT"] = "TavilySearch Search-Agent"

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# TavilySearch is the inbuilt optimised searching tool by Tavily itself
# Now we do not need the tool, since we are directly using the TavilySearch 

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello Job Aspirant!")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 AI Engineer Job Postings in Pune in February 2026.")})
    print(result)

if __name__ == "__main__":
    main()