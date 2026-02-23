from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.
    Args:
    query: The query to search for
    Returns:
    The result
    """
    print(f"Seaching for {query}")  # buffer display text
    return "Tokyo weather is sunny"


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    results = agent.invoke(
        {"messages": HumanMessage(content="What is the weather of Tokyo?")}
    )
    print(results)  # An llm response result contain various components, we mostly want the content attribute only


if __name__ == "__main__":
    main()
