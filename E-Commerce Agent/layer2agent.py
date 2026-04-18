# Layer2: We will not use lancghain objects this time, but OLLAMA Pyhton SDK
# We have al;ready installed the OLLAMA SDK inderectly when w installed langchain-ollama as a dependency in environment

from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools ---
# Plain Python functions — no @tool decorator here.
# Without @tool, LangSmith won't auto-detect these as tools, so we
# manually declare run_type="tool" so the trace shows them correctly.

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Manual JSON Tool Schema ---
# In LangChain, @tool auto-generated this from your type hints + docstring.
# Here we write it ourselves — this is the raw OpenAI-compatible JSON schema
# that Ollama (and most LLMs) natively understand.
#
# Each schema entry has 3 critical parts:
#   "name"        → exact identifier the LLM uses when it decides to call this tool
#   "description" → natural language hint that drives the model's tool selection
#   "parameters"  → JSON Schema defining the expected inputs;
#                   "required" prevents the model from omitting critical fields
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],  # LLM must always supply this — no guessing allowed
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],  # both fields mandatory — no partial calls
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.


# --- Traced Ollama wrapper ---
# ollama.chat() is a bare SDK call — LangSmith has no visibility into it.
# Wrapping it in @traceable(run_type="llm") makes every LLM call appear
# in the LangSmith trace, just like LangChain's instrumented wrappers did automatically.
# The tools schema travels with every request — Ollama has no bind_tools(),
# so unlike LangChain, tools are NOT pre-bound once; they are passed each time.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    # Plain dict mapping tool name → actual function.
    # Same purpose as Layer 1's tools_dict — O(1) lookup by name during the loop.
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # The agent's working memory — grows with every iteration.
    # Unlike Layer 1's typed objects (SystemMessage, HumanMessage),
    # these are raw dicts with a "role" key — the native format every LLM API uses underneath.
    # LangChain's message classes ultimately serialize down to exactly this.
    messages = [
        {
            "role": "system",       # sets the agent's persona and strict behavioral rules
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},  # the trigger that starts the reasoning chain
    ]

    # ReAct loop — each iteration is one full Reason → Act → Observe cycle.
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Call the LLM with the full message history.
        # Returns a ChatResponse object; the actual message lives in response.message.
        # Equivalent to llm_with_tools.invoke(messages) in Layer 1.
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message  # AIMessage equivalent — contains content + tool_calls

        tool_calls = ai_message.tool_calls  # list of tools the model decided to invoke this iteration

        # Empty tool_calls = model has enough info and is giving a final answer.
        # Same exit condition as Layer 1 — no tool needed → we're done.
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — one tool per iteration for clarity.
        # (Ollama supports parallel tool calls, but we enforce sequential execution here.)
        tool_call = tool_calls[0]

        # --- Ollama's tool call object structure ---
        # Unlike Layer 1 where tool_call was a plain dict (tool_call.get("name")),
        # the Ollama SDK returns a typed object that mirrors the JSON schema nesting:
        #
        #   tool_call
        #   └── .function
        #       ├── .name        → "get_product_price"
        #       └── .arguments  → {"product": "laptop"}
        #
        # The .function nesting directly reflects the "function" key in tools_for_llm above.
        tool_name = tool_call.function.name        # which tool the model chose
        tool_args = tool_call.function.arguments   # dict of arguments the model wants to pass

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Direct Python function call using ** unpacking.
        # tool_args is already a dict like {"product": "laptop"},
        # so **tool_args expands to get_product_price(product="laptop").
        # This replaces Layer 1's tool.invoke(tool_args) — same effect, zero overhead.
        observation = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observation}")

        # Append the AI's decision to the history so the model remembers what it just did.
        messages.append(ai_message)

        # Feed the tool result back as a "tool" role message.
        # Key differences from Layer 1's ToolMessage:
        #   - No tool_call_id needed — Ollama doesn't require ID matching (unlike OpenAI API).
        #   - content must be a string — cast observation even if it's a float (e.g. 1299.99).
        # On the next iteration, the model sees this result and reasons about the next step.
        messages.append(
            {
                "role": "tool",
                "content": str(observation),  # always stringify — "content" must be a string
            }
        )

    # Safety net — if we hit MAX_ITERATIONS without a final answer,
    # the model likely got stuck in a tool-calling loop. Fail gracefully.
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Ollama Agent (raw SDK)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")


# The complete documentation of the code has been attached in the file:
# "/layer2_ollama_agent_documentation.md" in the same folder.