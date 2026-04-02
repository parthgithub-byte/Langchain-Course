from dotenv import load_dotenv
load_dotenv()

import os

os.environ["LANGSMITH_PROJECT"] = "Pydantic Search-Agent"

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL for the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list,description="List of sources used to generate the answer")

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Helo Job Aspirant!")
    results = agent.invoke({
        "messages": HumanMessage(
            content="Help me list 3 best AI stack related courses on Udemy."
        )
    })
    print(results)
    
if __name__=="__main__":
    main()

# Here, the agent decides on the schema of the o/p using the function calling done with the primary schema provided as the object 
# of class AgentResponse. More of how the tool-calling is done by the agent is studied in the next module.