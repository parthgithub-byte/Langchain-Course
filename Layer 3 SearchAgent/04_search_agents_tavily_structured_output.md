# `jobsearch.py`, `searchagent.py`, `tavilyprovider.py`, `tavilypydantic.py`, `tavilysearch.py`, `jobsearch2.py`
## Layer 3: Prebuilt Agents (`create_agent`) + Web Search Tooling + Structured Output

## What these files are

All six files use LangChain's **prebuilt agent constructor**, `create_agent(model=..., tools=...)`, instead of the hand-rolled ReAct loop from Layers 1 and 2. `create_agent` internally implements the same reason → act → observe cycle you built manually before, but hides it behind a single call. Every file in this group answers a question that needs **live web data** the model doesn't have — so the point of the group is: *how do you give an agent internet access, and how do you shape what it hands back?*

The six files form a clear progression:

| File | What it adds/changes | Search tool | Structured output? |
|---|---|---|---|
| `jobsearch.py` | Baseline: custom search tool + `create_agent` | Hand-wrapped `TavilyClient` via `@tool` | No — raw message list |
| `searchagent.py` | Same pattern, different query/model; documents the raw LangSmith trace shape | Hand-wrapped `TavilyClient` via `@tool` | No |
| `tavilysearch.py` | Swaps the hand-written tool for LangChain's **prebuilt** `TavilySearch()` | `langchain_tavily.TavilySearch` | No |
| `tavilypydantic.py` | Adds a Pydantic schema so the agent returns typed `answer` + `sources` | `TavilySearch` | Yes — `response_format=AgentResponse` (default "tool strategy") |
| `tavilyprovider.py` | Same schema, but forces **native** structured output instead of the tool-calling hack | `TavilySearch` | Yes — `response_format=ProviderStrategy(schema=AgentResponse)` |
| `jobsearch2.py` | Same baseline pattern as `jobsearch.py`, but swaps the model provider to **OpenRouter** | Hand-wrapped `TavilyClient` via `@tool` | No |

Read in this order, the group teaches: *custom tool → prebuilt tool → typed output → optimized typed output → provider portability.*

---

## Shared foundation: `create_agent`

```python
from langchain.agents import create_agent
...
agent = create_agent(model=llm, tools=tools)
...
result = agent.invoke({"messages": HumanMessage(content="...")})
```

- `create_agent` is LangChain's batteries-included agent builder. Give it a model and a list of tools, and it wires up the same bind-tools → loop → tool-call → feed-back-result cycle from `layer1agent.py`, but you never see the loop code.
- The invocation shape is always `{"messages": [...]}` — a **list of messages**, matching the underlying chat message protocol seen throughout this project (`HumanMessage`, `SystemMessage`, etc. from `langchain_core.messages`).
- `result` is a state dict; without `response_format` it typically holds the full running `messages` list (see `searchagent.py`'s printed LangSmith trace below for what that looks like at each step). With `response_format` set (last two files), `result` becomes an instance of your Pydantic model directly.

---

## `jobsearch.py` — custom Tavily tool, no structured output

```python
from tavily import TavilyClient
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)
```
- `TavilyClient()` is the **raw Tavily SDK client** (not a LangChain integration) — you're calling Tavily's search API directly inside your own `@tool`-decorated function.
- Note the docstring uses **Google-style `Args:`/`Returns:`** formatting — this matters because `@tool` (and separately, Ollama's auto-schema feature seen in Layer 2) parses this format to build accurate parameter descriptions for the LLM.
- **Model choice matters here**: the trailing comment explicitly flags `gemini-3.6-flash` as required — "It is slower, but handles agentic workflow" — implying faster/cheaper Gemini variants were tried and didn't reliably drive multi-step tool use.
- The query — *"search for 3 job postings for an ai engineer using langchain in the Pune or Mumbai area on linkedin"* — demonstrates the agent doing multiple things behind the scenes with **one tool call type**: reformulating the request into a search query, calling `search`, then synthesizing structured markdown (headers, bullet lists, links) from unstructured search results, entirely from the system's default behavior (no explicit output formatting requested).

---

## `searchagent.py` — same pattern, illustrating the trace shape

```python
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    results = agent.invoke({"messages": HumanMessage(content="What is the weather of Tokyo?")})
    print(results)
```
- Nearly identical to `jobsearch.py`'s tool setup (same `@tool`-wrapped `TavilyClient`) and the same model (`gemini-3.6-flash`), but a simpler question — a single-fact weather lookup doesn't need the heavier agentic reasoning `jobsearch.py`'s multi-item job search required.
- The commented-out trace at the bottom is the most valuable part of this file — it shows **exactly what `create_agent` does internally**, message by message:
  1. **Human** → `"What is the weather of Tokyo?"`
  2. **AI** → decides to call `search`, with `query: "weather in Tokyo"` (the model reformulates the user's question into a good search query — it doesn't just forward the raw question)
  3. **Tool** → raw JSON blob returned by Tavily: multiple result objects, each with `title`, `url`, `content`, `score`, plus top-level `response_time` and `request_id`
  4. **AI** → final natural-language answer, synthesized from the tool's JSON (temperature, wind, humidity, local time), attributing this to `create_agent`'s automatic "read tool output → write final answer" step

This confirms `create_agent` is doing precisely the Reason → Act → Observe loop from Layers 1–2, just without exposing the `for` loop, `tool_calls` check, or message-appending code to you.

---

## `jobsearch2.py` — same custom-tool pattern, different model provider (OpenRouter)

```python
from langchain_openrouter import ChatOpenRouter
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
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
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for Automation Testing roles in Pune")})
    print(result)
```
- **Structurally identical** to `jobsearch.py`: same `TavilyClient()` + `@tool`-wrapped `search` function (down to the docstring shape), same `create_agent(model=llm, tools=tools)` call, same `{"messages": HumanMessage(...)}` invocation pattern, no `response_format`.
- The **only real change** is the model wrapper: `ChatOpenRouter(model="nvidia/nemotron-3.5-lightning:free")` in place of `ChatGoogleGenerativeAI(model="gemini-3.6-flash")`. This is the same "swap the model, keep the pipeline" lesson from `main1.py`/`main2.py` and `layer1agent.py`/`layer1agentGemini.py`, now applied one level up at the agent layer — proving `create_agent` is just as provider-agnostic as `.bind_tools()` and `init_chat_model()` were.
- **OpenRouter** is a routing layer that gives access to many different underlying models (here, NVIDIA's `nemotron-3.5-lightning`, on a free tier) through one unified API/interface — conceptually similar to why `init_chat_model()` existed in Layer 1: one consistent LangChain interface, swappable backend.
- The query itself (*"3 job postings for Automation Testing roles in Pune"*) is a variant of `jobsearch.py`'s LinkedIn/AI-engineer search — same task shape (structured multi-item job listing via search), different domain, again showing the agent pattern generalizes across query topics without code changes.
- No output formatting is applied here (no `response_format`), so like `jobsearch.py` and `searchagent.py`, `result` is the raw `create_agent` state — printed as-is rather than accessed via typed attributes.

---

## `tavilysearch.py` — prebuilt `TavilySearch` tool replaces the custom wrapper

```python
from langchain_tavily import TavilySearch
# TavilySearch is the inbuilt optimised searching tool by Tavily itself
# Now we do not need the tool, since we are directly using the TavilySearch

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)
```
- Where `jobsearch.py`/`searchagent.py` manually wrapped `TavilyClient` inside a `@tool` function, this file uses **`langchain_tavily.TavilySearch()`** directly as the tool object — no `@tool` decorator, no manual docstring, no manual `tavily.search(...)` call. It's LangChain's own maintained integration, pre-optimized for agent use (better default result formatting, built-in schema).
- This is the same "hand-written vs. framework-provided" tradeoff seen in Layer 2 (`layer2agent.py`'s manual JSON schema vs. auto-generation) — except here it goes the *other* direction: moving from manual back to prebuilt.

```python
os.environ["LANGSMITH_PROJECT"] = "TavilySearch Search-Agent"
# This ignores whatever LANGSMITH_PROJECT was in the .env
```
- Setting `LANGSMITH_PROJECT` **in code, after `load_dotenv()`**, deliberately overrides any value from `.env` — this routes this specific script's traces into their own named LangSmith project, separate from other scripts' traces, which is useful once you have several experiments/agents you want to inspect independently. This pattern is reused in `tavilypydantic.py` (`"Pydantic Search-Agent"`) and `tavilyprovider.py` (`"Provider Strategy"`).

---

## `tavilypydantic.py` — typed structured output via `response_format`

```python
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL for the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
```
- Two **Pydantic models** define the exact output shape: a top-level `answer` string plus a list of `Source` objects (each just a `url`).
- Passing `response_format=AgentResponse` directly (a raw Pydantic class) tells `create_agent` to use its **default strategy** for enforcing structure — under the hood, this is the "tool-calling hack": the agent is given an internal pseudo-tool that matches your schema and is essentially told "call this to submit your final answer."
- Payoff: `results` is no longer a raw message dict — it's an `AgentResponse` **instance**, so `results.answer` and `results.sources` (each a `Source` with `.url`) work directly, no manual parsing of message content.
- Comment in the file: *"the agent decides on the schema of the o/p using the function calling done with the primary schema... More of how the tool-calling is done by the agent is studied in the next module"* — explicitly flags this as the "old"/default mechanism, setting up the contrast with the next file.

---

## `tavilyprovider.py` — the same schema, but via `ProviderStrategy`

```python
from langchain.agents.structured_output import ProviderStrategy
...
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ProviderStrategy(schema=AgentResponse)
)
```
- Identical `Source`/`AgentResponse` Pydantic schema to `tavilypydantic.py` — the **only** change is wrapping it in `ProviderStrategy(schema=...)` instead of passing the raw class.
- This switches from the tool-calling hack to the model provider's **native structured-output API** (Gemini supports this directly) — the model's token generation is constrained to match your JSON schema at the API level, rather than pretending to call a tool named something like `OutputFormatter`.
- The file's trailing docstring lays out exactly why this is preferred when the provider supports it:
  - **Zero "hack" overhead** — no pretend tool-calling reasoning step
  - **Cheaper and faster** — fewer tokens spent, since the "I should call a tool now" internal reasoning is skipped
  - **Higher reliability** — schema enforcement happens at the API level, so it's far less prone to hallucinated fields or broken JSON than the tool-calling approach
- Usage is unchanged from `tavilypydantic.py`: `results.answer`, and iterate `results.sources` for `.url` — the *interface* is identical; only the *mechanism* generating it differs. This is the same lesson Layer 2 taught (framework vs. protocol) applied to output formatting instead of tool calling.

---

## Interfile relations & why the approach changes across the group

1. **`jobsearch.py` → `searchagent.py`**: same custom-tool pattern proven on two different query types (a multi-item structured search vs. a single-fact lookup), with model choice adjusted to match task complexity.
2. **`searchagent.py` → `tavilysearch.py`**: the hand-rolled `@tool` wrapper around `TavilyClient` is replaced by LangChain's dedicated `TavilySearch` integration — less code, same capability, tuned specifically for agent consumption.
3. **`tavilysearch.py` → `tavilypydantic.py`**: adds `response_format` so output isn't just readable prose but a **typed, program-consumable object** — necessary once you want to do something with the agent's output besides printing it (e.g. render sources as clickable links, store answers in a database).
4. **`tavilypydantic.py` → `tavilyprovider.py`**: same schema, but swaps *how* structure is enforced — from the older, universally-compatible tool-calling hack to the provider-native mechanism, once the model (Gemini) supports it — trading a small amount of portability for speed, cost, and reliability.

## Concept summary table

| Concept | Where introduced | Purpose |
|---|---|---|
| `create_agent(model, tools)` | All 5 files | Prebuilt ReAct-style agent; hides the manual loop from Layers 1–2 |
| `TavilyClient()` + `@tool` | `jobsearch.py`, `searchagent.py` | Manually wrap a 3rd-party search SDK as an agent tool |
| `langchain_tavily.TavilySearch()` | `tavilysearch.py` onward | Prebuilt, agent-optimized search tool — no manual wrapping |
| `os.environ["LANGSMITH_PROJECT"] = ...` | `tavilysearch.py`, `tavilypydantic.py`, `tavilyprovider.py` | Route each script's trace into its own named LangSmith project, overriding `.env` |
| `response_format=SomePydanticModel` | `tavilypydantic.py` | Default structured output via internal tool-calling hack |
| `response_format=ProviderStrategy(schema=...)` | `tavilyprovider.py` | Native provider structured output — faster, cheaper, more reliable |
| `agent.invoke({"messages": HumanMessage(...)})` | All 5 files | Standard entry point; input is always a list of chat messages |

## Note on images
No image files or references to screenshots/diagrams appear in the comments of any of these five files, so none are included here.