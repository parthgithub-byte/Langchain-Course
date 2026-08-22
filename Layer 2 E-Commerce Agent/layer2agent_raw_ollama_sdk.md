# `layer2agent.py` — Layer 2: The Same Agent, No LangChain (Raw Ollama SDK)

## What this file is

This rebuilds the **exact same shopping agent** from `layer1agent.py` / `layer1agentGemini.py` — same two tools, same system prompt, same ReAct loop shape, same example question — but with **every LangChain abstraction stripped out**, using only the raw **Ollama Python SDK**. The ReAct *logic* (reason → act → observe → repeat) is unchanged; only the *plumbing* underneath changes. This makes it the most "honest" view of what an agent really is at the protocol level.

A companion file, `layer2_ollama_agent_documentation.md`, already documents the conceptual differences from Layer 1 in table form — this doc folds that comparison in directly, next to the actual code.

---

## Setup

```python
from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
```
Note: `ollama` here is imported directly — no `langchain_ollama` wrapper. The comment in the file points out the SDK was already installed indirectly as a dependency of `langchain-ollama` in Layer 1's environment.

---

## Difference 1 — Manual JSON schema instead of `@tool`

Layer 1 got a JSON schema for free from `@tool` reading your type hints + docstring. Here, you write it by hand:

```python
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
                "required": ["product"],
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
                    "discount_tier": {"type": "string", "description": "The discount tier: 'bronze', 'silver', or 'gold'"},
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]
```
This is the **OpenAI-compatible JSON schema format** most LLM providers (including Ollama) understand natively. Three parts matter:
- `"name"` — exact identifier the model uses to request the tool.
- `"description"` — drives the model's tool-*selection* accuracy; this is literally the model's only understanding of what the tool does.
- `"parameters"` / `"required"` — JSON Schema defining and mandating inputs, preventing the model from omitting critical fields.

> **Shortcut that exists but isn't used here:** Ollama *can* auto-generate schemas from plain functions (`tools_for_llm = [get_product_price, apply_discount]`), but only if docstrings follow **Google docstring format** with an `Args:` section. The manual version is kept deliberately, to show what `@tool` was hiding.

---

## Tools — plain functions, manually traced

```python
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
```
Same logic as Layer 1's tools — but now they're **bare Python functions**, not `StructuredTool` objects. Without `@tool`, LangSmith has no automatic way to know these are tool calls, so `run_type="tool"` is passed explicitly to `@traceable` to keep them correctly labeled in the trace.

---

## Difference 2 — Manual tracing everywhere

Layer 1's LangSmith integration was automatic (LangChain's wrappers are pre-instrumented). Here, tracing has to be applied by hand in **three separate places**:
- `@traceable(run_type="tool")` on each tool function (above)
- `@traceable(name="Ollama Chat", run_type="llm")` on the model-call wrapper (below)
- `@traceable(name="Ollama Agent Loop")` on `run_agent` itself

```python
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
```
`ollama.chat()` is a bare SDK call with zero LangSmith awareness — wrapping it is the only way to make it show up in a trace at all.

---

## Difference 3 — No `bind_tools()`: schema passed every call

Layer 1 bound tools to the model **once** via `llm.bind_tools(tools)`; every later `.invoke()` automatically knew about them. Ollama's raw SDK has no equivalent — `tools_for_llm` must be passed **explicitly on every single call**, as seen above (`tools=tools_for_llm`). This mirrors how tool calling actually works at the protocol level: the full tool definitions travel with every request.

---

## The agent loop

```python
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }
```
Same purpose as Layer 1's `tools_dict` — O(1) lookup by tool name — but now mapping directly to plain functions instead of `StructuredTool` objects.

### Difference 4 — Messages as plain dicts

```python
messages = [
    {"role": "system", "content": (...)},   # identical guardrail rules to Layer 1, verbatim
    {"role": "user", "content": question},
]
```
Layer 1 used typed objects (`SystemMessage`, `HumanMessage`, `ToolMessage`). Here it's raw dicts with a `"role"` key — the actual **wire format** every LLM chat API uses underneath. LangChain's message classes ultimately serialize down to exactly this; Layer 2 just skips the intermediate objects. The system prompt content is unchanged from Layer 1 — same strict rules — proving the agent's *reasoning behavior* is independent of the framework.

### The loop itself

```python
for iteration in range(1, MAX_ITERATIONS + 1):
    response = ollama_chat_traced(messages=messages)
    ai_message = response.message
    tool_calls = ai_message.tool_calls

    if not tool_calls:
        return ai_message.content
```
- `ollama.chat()` returns a `ChatResponse` object; the actual message (with any tool call decisions) lives at `response.message` — the equivalent of Layer 1's `AIMessage`. Variable is deliberately named `ai_message` to keep the same mental model across both layers.
- Same exit condition as Layer 1: empty `tool_calls` means the model is ready to answer.

### Difference 5 — Attribute access instead of dict access

```python
tool_call = tool_calls[0]
tool_name = tool_call.function.name        # Layer 1: tool_call.get("name")
tool_args = tool_call.function.arguments   # Layer 1: tool_call.get("args", {})
```
Ollama's SDK returns tool calls as **typed objects**, not dicts. The structure mirrors the JSON schema nesting you wrote earlier:
```
tool_call
└── .function
    ├── .name        → "get_product_price"
    └── .arguments   → {"product": "laptop"}
```
The `.function` attribute directly reflects the `"function"` key used in `tools_for_llm`.

### Difference 6 — Direct function call instead of `.invoke()`

```python
tool_to_use = tools_dict.get(tool_name)
if tool_to_use is None:
    raise ValueError(f"Tool '{tool_name}' not found")

observation = tool_to_use(**tool_args)   # Layer 1: tool_to_use.invoke(tool_args)
```
`tool_args` is a plain dict (e.g. `{"product": "laptop"}`); `**tool_args` unpacks it into keyword arguments. Same effect as `.invoke()`, but idiomatic Python with zero LangChain overhead.

### Difference 7 — Result feedback as a plain dict

```python
messages.append(ai_message)
messages.append({
    "role": "tool",
    "content": str(observation),   # must always be a string, even for a float like 1299.99
})
```
Two key differences from Layer 1's `ToolMessage(content=..., tool_call_id=...)`:
1. **No `tool_call_id`** — unlike the OpenAI API (which Layer 1's `ToolMessage` protocol assumes), Ollama's tool-result format does not require ID matching between a call and its result.
2. The observation is explicitly cast to `str`, since `"content"` must always be a string.

### Safety net

```python
print("ERROR: Max iterations reached without a final answer")
return None
```
Identical purpose to Layer 1.

---

## Layer 1 vs Layer 2 — full comparison table

| Concept | Layer 1 (LangChain) | Layer 2 (raw Ollama SDK) |
|---|---|---|
| Tool schema | Auto-generated by `@tool` | Hand-written JSON |
| LLM call | `llm_with_tools.invoke()` | `ollama.chat(..., tools=...)` |
| Tools attached | Once, via `bind_tools()` | Every call, via `tools=tools_for_llm` |
| Messages | Typed objects (`SystemMessage`, etc.) | Plain dicts with `"role"` |
| Tool call parsing | `tool_call.get("name")` (dict) | `tool_call.function.name` (object) |
| Tool execution | `tool.invoke(args)` | `tool(**args)` |
| Tool result message | `ToolMessage(tool_call_id=...)` | `{"role": "tool", "content": ...}` — no ID |
| LangSmith tracing | Automatic | Manual `@traceable` on every function |

**What doesn't change at all:** the ReAct loop shape (reason → select tool → execute → observe → repeat), the system prompt's guardrail rules, and the two tools' business logic. The agent's *thinking* is framework-agnostic — only the *wiring* differs. This is the core lesson Layer 2 is built to teach.

---

## Interfile relation

- Directly rebuilds `layer1agent.py`'s agent (same tools, same prompt, same question: *"What is the price of a laptop after applying a gold discount?"*).
- Its own conceptual write-up already exists as `layer2_ollama_agent_documentation.md` — this file matches that write-up to the actual code, side by side.
- No image references in this file's comments (unlike `layer1agent.py`, which pointed to a debugger screenshot folder).
