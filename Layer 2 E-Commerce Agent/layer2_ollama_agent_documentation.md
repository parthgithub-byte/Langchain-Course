# Layer 2: The Agent Loop — Rebuilding with the Ollama Python SDK (No LangChain)

This layer strips away all LangChain abstractions and rebuilds the exact same shopping agent using only the **Ollama Python SDK**. The logic and the ReAct loop are identical to Layer 1 — what changes is *everything underneath*: how tools are declared, how the model is called, how responses are parsed, and how tool results are fed back. Every place where Layer 1 did something automatically, Layer 2 does it manually. This makes Layer 2 the most honest view of what an agent actually is at the protocol level.

---

## The First Major Difference: Manual JSON Schema vs. `@tool`

In Layer 1, LangChain's `@tool` decorator silently read your function's type hints and docstring and generated a JSON schema for the LLM. Here, there is no decorator to do that — you write the schema yourself:

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
                    }
                },
                "required": ["product"],
            },
        },
    },
    ...
]
```

This is the **OpenAI-compatible JSON schema format** that Ollama (and most LLMs) natively understand. The schema has three critical parts:

- `"name"` — the exact function identifier the LLM will use to request a tool call.
- `"description"` — natural language that helps the model decide *when* to use this tool. The quality of this description directly impacts the model's tool selection accuracy.
- `"parameters"` — a JSON Schema object describing the expected inputs. The `"required"` array tells the model which arguments it *must* provide, preventing it from omitting critical fields.

This is not busywork — this schema is the LLM's entire understanding of your tool. `@tool` generated exactly this from your docstring. Now you own it.

> **Note on the shortcut:** As the Ollama docs confirm, the Python SDK *can* auto-parse plain functions into tool schemas — but only if your docstrings follow **Google docstring format** (with an `Args:` section). The manual JSON version is kept here deliberately, so you see what the decorator was hiding.

---

## Tracing Without LangChain: `@traceable` Manually Applied

In Layer 1, LangChain's integration with LangSmith was largely automatic. Here, we have three separate places to apply `@traceable` manually:

```python
@traceable(run_type="tool")
def get_product_price(product: str) -> float: ...

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float: ...

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages): ...
```

Each decorator tells LangSmith *what kind of operation* this is — `"tool"` for tool executions, `"llm"` for model calls. This is the manual bookkeeping that LangChain's instrumented wrappers handled automatically. The `ollama_chat_traced` wrapper exists purely because `ollama.chat()` is a bare SDK call with no LangSmith awareness — you need to wrap it in a traceable function to make it visible in the trace.

---

## No `bind_tools`: Passing the Schema Directly at Call Time

In Layer 1, tools were bound to the model once upfront with `llm.bind_tools(tools)` — from that point on, every invocation knew about the tools automatically. Here, there is no `bind_tools`. Instead, the `tools_for_llm` schema list is passed directly into every `ollama.chat()` call:

```python
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
```

This is the Ollama SDK's equivalent of what `bind_tools` does — except you do it explicitly each time. Per the official Ollama documentation, this is how tool calling works at the protocol level: the tool definitions travel with every request so the model always has the full context of what it can invoke.

---

## The Messages List: Plain Dicts Instead of Message Objects

In Layer 1, the messages list used LangChain's typed objects — `SystemMessage`, `HumanMessage`, `ToolMessage`. Here, everything is plain Python dictionaries with a `"role"` key:

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": question},
]
```

This is the raw **chat message format** that underlies every LLM API — LangChain's message classes ultimately serialize down to exactly this. The `"role"` field can be `"system"`, `"user"`, `"assistant"`, or `"tool"`. The system prompt content is identical to Layer 1 — same strict rules, same behavioral constraints — because the agent's logic hasn't changed, only the plumbing.

---

## Inside the ReAct Loop: `response.message` vs. `ai_message`

```python
response = ollama_chat_traced(messages=messages)
ai_message = response.message
```

`ollama.chat()` returns a `ChatResponse` object. The actual message content — including any tool call decisions — lives in `response.message`. This is equivalent to the `AIMessage` returned by LangChain's `.invoke()` in Layer 1. The variable is renamed `ai_message` to preserve the same mental model across both layers.

---

## The Third Major Difference: Attribute Access vs. Dict Access

This is where the surface difference between the two layers is most visible. In Layer 1, a tool call was a dictionary and you used `.get()` to safely extract fields:

```python
# Layer 1 — dict access
tool_name = tool_call.get("name")
tool_args = tool_call.get("args", {})
```

In Layer 2, the Ollama SDK returns tool calls as **Python objects with a `.function` attribute**:

```python
# Layer 2 — attribute access
tool_name = tool_call.function.name
tool_args = tool_call.function.arguments
```

The Ollama SDK models a tool call as `ToolCall → function → { name, arguments }`, mirroring the JSON schema structure you defined earlier. This is a direct reflection of how the Ollama API represents tool calls internally — the `"function"` nesting in the schema corresponds directly to the `.function` attribute access here.

---

## The Fourth Major Difference: Direct Function Call vs. `.invoke()`

In Layer 1, tools were `StructuredTool` objects and were executed via LangChain's `.invoke()` method:

```python
# Layer 1
observation = tool_to_use.invoke(tool_args)
```

Here, the tools in `tools_dict` are plain Python functions. You call them using Python's `**kwargs` unpacking:

```python
# Layer 2
observation = tool_to_use(**tool_args)
```

`tool_args` is already a dictionary of argument names to values (e.g., `{"product": "laptop"}`). The `**` operator unpacks it directly into the function's keyword arguments. This is idiomatic Python — clean, zero-overhead, and exactly what `.invoke()` was doing internally.

---

## Feeding the Result Back: The `"tool"` Role Dict

In Layer 1, the tool result was wrapped in a `ToolMessage` object with a `tool_call_id` to match the response back to its originating request:

```python
# Layer 1
messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
```

In Layer 2, it's a plain dict with `"role": "tool"`:

```python
# Layer 2
messages.append({"role": "tool", "content": str(observation)})
```

Two things to notice here. First, no `tool_call_id` — unlike the OpenAI API, Ollama's tool message format does not require matching IDs. As shown in the official Ollama documentation, the tool result message simply carries `role`, `tool_name`, and `content`. Second, the observation is cast to `str` — because the `"content"` field must always be a string, even when the tool returns a float like `1299.99`.

---

## What This Layer Teaches You

By removing LangChain, Layer 2 makes the following things explicit that Layer 1 kept implicit:

| Concept | Layer 1 (LangChain) | Layer 2 (Ollama SDK) |
|---|---|---|
| Tool schema | Auto-generated by `@tool` | Hand-written JSON |
| LLM call | `llm_with_tools.invoke()` | `ollama.chat(..., tools=...)` |
| Tool call parsing | `tool_call.get("name")` (dict) | `tool_call.function.name` (object) |
| Tool execution | `tool.invoke(args)` | `tool(**args)` |
| Tool result message | `ToolMessage(id=...)` | `{"role": "tool", "content": ...}` |
| LangSmith tracing | Automatic | Manual `@traceable` on each fn |

The ReAct loop itself — reason, select tool, execute, observe, repeat — is completely unchanged. The agent's *thinking* is framework-agnostic. Only the *wiring* is different.
