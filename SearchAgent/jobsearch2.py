from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query:str)->str:
    """
    A tool that searches over the internet
    Args:
        query: The query to seach for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenRouter(model="nvidia/nemotron-3.5-lightning:free")
tools = [search]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Program starts now...")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for Automation Testing roles in Pune")})
    print(result)

if __name__ == "__main__":
    main()