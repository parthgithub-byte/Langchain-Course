# `layer1agent.py` & `layer1agentGemini.py` — Layer 1: The ReAct Agent Loop (LangChain)

## What these files are

This is the project's first real **agent**: instead of one prompt → one answer, the LLM can call tools, look at results, and decide what to do next — a manual **ReAct loop** (Reason → Act → Observe → repeat), built with LangChain's `.bind_tools()`.

Both files are **identical in structure and logic**. The only difference is which model powers the loop:

| | `layer1agent.py` | `layer1agentGemini.py` |
|---|---|---|
| Model | `qwen3:1.7b` via **Ollama** (local) | `gemini-3.1-flash-lite-preview` via **Google GenAI** (cloud) |
| Init call | `init_chat_model(f"ollama:{MODEL}", temperature=0)` | `init_chat_model("google_genai:gemini-3.1-flash-lite-preview", temperature=0)` |
| Observed latency | ~223 sec | ~161 sec (after retries — free tier rate-limited on first attempt) |
| Final answer shape | Plain string | List of content blocks with a `signature` field (Gemini-specific metadata) |

This pairing again demonstrates provider-agnosticism: swapping the model string in `init_chat_model(...)` is the *only* change needed to move the exact same agent from a local model to a cloud model.

---

## Setup

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
```
- `init_chat_model` — a provider-agnostic model loader. You give it a string like `"ollama:qwen3:1.7b"` or `"google_genai:gemini-3.1-flash-lite-preview"` and it figures out which integration to instantiate. This avoids importing a different class (`ChatOllama`, `ChatGoogleGenerativeAI`, ...) per provider, unlike `main1.py`/`main2.py`.
- `MAX_ITERATIONS` — a hard safety cap so the ReAct loop can't run forever if the model keeps calling tools without converging on an answer.
- `@traceable` (from `langsmith`) — wraps the whole `run_agent` function so its execution shows up as a trace in LangSmith (Anthropic-unrelated observability tool for LLM apps).

---

## Tools

```python
@tool
def get_product_price(product: str) -> float:
    """Looks up the price of a product in the catalog"""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
```
- The `@tool` decorator inspects the function's **type hints** and **docstring** and auto-generates a JSON schema the LLM can understand and call. You never write that schema by hand here (contrast this with Layer 2, where you do).
- Internally, `get_product_price` and `apply_discount` become `StructuredTool` objects, not plain functions.
- `tools_dict = {t.name: t for t in tools}` — a dictionary comprehension mapping tool name → tool object, so the loop can look up `"get_product_price"` → the actual callable in O(1).

---

## Binding tools to the model

```python
llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
llm_with_tools = llm.bind_tools(tools)
```
`bind_tools()` permanently attaches the tool schemas to this model instance. From here on, **every** call to `llm_with_tools.invoke(...)` automatically includes the tool definitions — you don't need to pass them again each time (contrast with Layer 2's Ollama SDK, which requires passing `tools=` on every single call).

---

## The system prompt (identical in both files)

```python
SystemMessage(content=(
    "You are a helpful shopping assistant."
    "You have access to a product catalog tool and a discount tool.\n\n"
    "STRICT RULES you must follow these exactly:\n"
    "1. NEVER guess or assume any product price. You MUST call get_product_price first...\n"
    "2. Only call apply_discount AFTER you have received a price from get_product_price...\n"
    "3. NEVER calculate the discount yourself mathematically. Always use the apply_discount tool.\n"
    "4. If the user does not specify a discount tier, ask them which tier to use..."
))
```
This is a **guardrail prompt** — it forces the model to rely on tools instead of hallucinating numbers or doing arithmetic itself, which LLMs are notoriously unreliable at. This exact same prompt text is reused verbatim in Layer 2, proving the *agent's behavior* doesn't depend on the framework — only the plumbing around it does.

---

## The ReAct loop, line by line

```python
for iteration in range(1, MAX_ITERATIONS):
    ai_message = llm_with_tools.invoke(messages)
    tool_calls = ai_message.tool_calls

    if not tool_calls:
        return ai_message.content   # model is done — no tool needed, give the final answer
```
- Each iteration = one full reasoning step.
- `ai_message.tool_calls` is a list — empty means the model thinks it already has enough information to answer directly.

```python
    tool_call = tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")
```
- Only the **first** tool call is processed per iteration (even though the model could request several in parallel) — kept simple for clarity/teaching purposes.
- `tool_call` here is a **plain dictionary**, so `.get(...)` is used defensively. `.get("args", {})` specifically guards against a `KeyError` if the model's response omits the `"args"` key entirely (can happen with smaller/local models like `qwen3:1.7b`).

```python
    tool_to_use = tools_dict.get(tool_name)
    if tool_to_use is None:
        raise ValueError(f"Tool {tool_name} not found.")
    observation = tool_to_use.invoke(tool_args)
```
- Tool lookup by name, then executed via `.invoke()` — because these are `StructuredTool` objects (LangChain-wrapped), not bare functions. This is the `Runnable` interface again, same `.invoke()` pattern used for the model itself.

```python
    messages.append(ai_message)
    messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
```
- The AI's own decision is appended to history so it "remembers" what it just did.
- The tool's result is wrapped in a `ToolMessage` — and critically, tagged with `tool_call_id` so the model can match this result back to the specific call it made (this ID-matching is an OpenAI-style protocol requirement; Layer 2 with raw Ollama does *not* need this — see that doc).

```python
print("ERROR: Max iterations reached without a final answer")
return None
```
Safety net if the loop never converges.

---

## Example trace (from `layer1agent.py`, `qwen3:1.7b`)

```
Question: What is the price of a laptop after applying a gold discount?

--- Iteration 1 ---
 [Tool Selected] get_product_price with args: {'product': 'laptop'}
   [Tool Result: 1299.99]

--- Iteration 2 ---
 [Tool Selected] apply_discount with args: {'price': 1299.99, 'discount_tier': 'gold'}
   [Tool Result: 1000.99]

--- Iteration 3 ---
Final Answer: The original price of the laptop was $1299.99.
After applying the gold discount, the final price is $1000.99.
```
This confirms the guardrail prompt worked: the model never guessed the price or did the math itself — it called both tools in the correct order.

In `layer1agentGemini.py`, the same flow runs, but the final answer arrives as a **list of content blocks** with an internal `signature` field — a Gemini-specific detail (used for response verification/caching), not something you'll see with Ollama models.

> **Note on images:** the original code comments reference a local `images` folder showing the `tool_call` object structure in a debugger. No image files were included in the upload, so they aren't reproduced here — the structure they'd show is documented below instead.

---

## `tool_call` object structure (dict form, Layer 1)

```python
tool_call = {
    "id": "326ac9cf....",
    "name": "get_product_price",
    "args": {"product": "laptop"},
}
```
Three fields matter: `id` (for `ToolMessage` matching), `name` (for `tools_dict` lookup), `args` (kwargs passed into the tool). This dict shape is exactly what `.get()` calls in the loop are built to safely unpack.

---

## Concept summary table

| Step | Concept | Purpose |
|---|---|---|
| `init_chat_model(...)` | Provider-agnostic loader | Swap models by changing a string, not an import |
| `@tool` | Auto schema generation | Type hints + docstring → JSON schema, no manual work |
| `bind_tools()` | Persistent tool binding | Tools attached once, available on every `.invoke()` |
| `SystemMessage` guardrails | Prompt engineering | Forces tool use over hallucination/math |
| `tool_calls` empty check | Loop exit condition | Distinguishes "needs a tool" vs "has the answer" |
| `tool_call.get(...)` | Defensive dict access | Avoids `KeyError` if model omits a field |
| `ToolMessage(tool_call_id=...)` | Result feedback | Matches a tool's output back to its specific call |
| `MAX_ITERATIONS` | Safety net | Prevents infinite tool-calling loops |

## Why this matters for later files
This is "Layer 1" specifically because Layer 2 (`layer2agent.py`) rebuilds this *exact same agent* — same tools, same system prompt, same loop shape — but strips out every LangChain convenience (`@tool`, `bind_tools`, `ToolMessage`, `.invoke()`) to show what's happening underneath at the raw SDK/protocol level.
