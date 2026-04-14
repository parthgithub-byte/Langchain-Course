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

    # the init_chat_model simplify the model declaration w/o any specific langchain imports 
    # with just the provider name as the necessity (may also skip if from specific family clearly)
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 100)

    # brain of the agent:
    messages = [
        # the instructions
        SystemMessage(
            content = (
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES  you must follow these eaxactly:\n"
                "1. NEVER guess or assume any product price."
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received"
                "a price from get_product_price. Pass the exact price "
                "received by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate the discount yourself mathematically. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        ),
        # the input
        HumanMessage(
            content=question
        )
    ]

    # The ReAct loop: Iterating from 1 to max no of iterations
    for iteration in range(1, MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        # invokes the llm using the messages having instructions and the input, returns a big nested dictionary output.

        tool_calls = ai_message.tool_calls
        # list of the tool calls (in that iteration) associated with result
        
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        # thus, if no tool call was present in that iteration, the ai_message result was inicated to the answer, since no too call were necessary 
        # (refer to the ReAct diagram)

        # Executing the first decided tool in the AIMessage only.
        # That is because, an agent may decide on multiple tools and call them in a step parallely but for the sake of clarity, we will carry out one at a time

        


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")


# If you go and apply a breakpoint on the if not tool_calls, the break will be applied right at the first iteration, 
# there you can see in the debugger that, the ai_message has the object "tool_calls" as we intended and it consists of the tool "get_product_price", which is 
# the appropriate tool for the first iteration, when product name and tier are passed.




# Some clarifications in the above code:
# When we declare:
# tools = [get_product_price, apply_discount], it is not a list of any simple functions but structured tools identified by the @tool decorators
# Internally, they are stored as:
# [
#   StructuredTool(name="get_product_price"),
#   StructuredTool(name="apply_discount")
# ]
# Thus, we store the tools_dict as:
# tools_dict = {t.name: t for t in tools}    This is called dictionary comprehension (It stores the "name of tool":tool pair)
# Internally it stores as:
# {
#   "get_product_price": get_product_price,
#   "apply_discount": apply_discount
# }
# We later use this tools_dict for its direct direction through tool name to the tool
