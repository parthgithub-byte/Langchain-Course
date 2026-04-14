from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
# custom limit of iterations
MODEL = "qwen3:1.7b"

# Tools

@tool
def get_product_price(product:str)->float:
    """Looks up the price of a product in the catalog"""
    print(f"   >> Executing get_product_price[product= '{product}']")
    prices = {"laptop":1299.99, "headphones":149.95, "keyboard":89.50}
    return prices.get(product,0)

@tool
def apply_discount(price:float, discount_tier:str)->float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"   >>Executing apply_discount(price={price}, discount_tier = '{discount_tier}')")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1-discount/100), 2)


# Agent loop

@traceable(name="LangCgain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 100)

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")