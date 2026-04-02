from dotenv import load_dotenv
load_dotenv()

import os

os.environ["LANGSMITH_PROJECT"] = "Provider Strategy"

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ProviderStrategy

class Source(BaseModel):
    """Schema for a scource used by the agent"""
    url:str = Field(description="The url for the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tools = [TavilySearch()]
agent = create_agent(
    model = llm, 
    tools = tools, 
    response_format=ProviderStrategy(schema=AgentResponse)
)

def main():
    print("Hello Aspirant!")
    results = agent.invoke({
        "messages" : HumanMessage(
            content="Help me find 3 AI Engineer job roles in Mumbai for Freshers."
        )
    })

    # Raw result, but it is in the desired dictionary keys "answer" and "sources" this time 
    print(results)

    print("\nAnswer:\n", results.answer)

    print("\nSources:")
    for src in results.sources:
        print("-", src.url)

if __name__ =="__main__":
    main()

"""
Why ProviderStrategy beats the Default Tool Strategy?
Before native JSON schemas existed, the only way to force an LLM to output reliable JSON was a clever hack: telling the LLM it had a tool called OutputFormatter and forcing it to use it.

Now that models like Gemini have native structured output APIs, the ProviderStrategy is superior because:

Zero "Hack" Overhead: The model doesn't have to pretend it's using a tool. It just natively constrains its token generation to your schema.

Cheaper and Faster: It uses fewer tokens because it skips the internal "I should call a tool now" reasoning steps.

Higher Reliability: Native schema enforcement at the API level (which is what ProviderStrategy taps into) is significantly less prone to hallucinating fields or breaking JSON structure than the old tool-calling method.
"""