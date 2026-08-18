# `main1.py` & `main2.py` — LangChain Basics: Prompt → Model → Chain

## What these files are

Both files do the **exact same thing**: take a hard-coded biography of Salman Khan, feed it into a `PromptTemplate`, pipe it into a chat model, and print a summary + two facts.

The only difference between them is **which model answers the prompt**:

| | `main1.py` | `main2.py` |
|---|---|---|
| Model provider | Google Gemini (cloud) | Ollama (local) |
| Import | `from langchain_google_genai import ChatGoogleGenerativeAI` | `from langchain_ollama import ChatOllama` |
| Model used | `gemini-3.0-flash` | `gemma3:270m` |
| Needs API key? | Yes (`GOOGLE_API_KEY` via `.env`) | No — runs locally |

This pairing exists to demonstrate that **LangChain's chain logic is provider-agnostic** — you can swap `ChatGoogleGenerativeAI` for `ChatOllama` and nothing else in the pipeline changes. This is the first appearance of a theme that recurs across the whole project: *the framework's abstractions stay the same, only the "engine" underneath changes.*

---

## Full flow (identical in both files)

```python
load_dotenv()
```
Loads secrets from a `.env` file into environment variables. `main1.py` needs this for `GOOGLE_API_KEY`. `main2.py` doesn't strictly need it (Ollama is local, no key), but keeps the call for consistency.

```python
information = """... Salman Khan biography ..."""
```
A plain Python string used as **grounding context**. Since LLMs have a knowledge cutoff, this text is handed to the model directly so it doesn't need to "know" or guess anything — it just reads and summarizes what's given.

```python
summary_template = """
Given the context {information} about a person, I want you to create:
1. Short summary
2. Two interesting facts about the person
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)
```
- `{information}` is a placeholder.
- `PromptTemplate` turns the raw string into a reusable, structured object LangChain understands.
- **Why not just use an f-string?** Because `PromptTemplate` objects can be saved, shared, and reused across a larger app — an f-string can't be composed into a chain.

```python
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.6-flash")   # main1.py
llm = ChatOllama(temperature=0, model="gemma3:270m")                    # main2.py
```
This is the only line that meaningfully differs between the two files.
- `temperature=0` → deterministic, factual output. No creative liberties — important for summarizing a biography without hallucination.
- The **model wrapper** is what turns a generic Python object into something LangChain's chain machinery can call `.invoke()` on.

```python
chain = summary_prompt_template | llm
```
The **pipe operator (`|`)** — this is LCEL (LangChain Expression Language). See the deep-dive below; it's the same mechanism in both files.

```python
response = chain.invoke(input={"information": information})
print(response.content)
```
- `invoke()` runs the pipeline: fills the template → sends the full string to the model → gets back a response object.
- `response` is a rich `AIMessage` object (contains token counts, stop reason, etc.) — `.content` extracts just the text.

---

## Deep dive: the `|` (pipe) operator — LCEL

This is the same in every LangChain file in this project, so it's worth understanding once, deeply.

- In vanilla Python, `|` is the bitwise OR operator — but Python allows **operator overloading**.
- Every LangChain component (`PromptTemplate`, `ChatGoogleGenerativeAI`, `ChatOllama`, etc.) inherits from a base class called `Runnable`, which implements `__or__`.
- So `prompt | model` is really Python executing `prompt.__or__(model)`, which builds a `RunnableSequence` — an object that knows: *"run `prompt` first, then feed its exact output into `model`."*

```python
# What you write:
chain = prompt | model
# What Python does:
chain = prompt.__or__(model)
```

**Why this design (the "Unix philosophy")**: it mirrors shell pipes (`cat file | grep "error"`) — data flows left to right, which is far more readable than nested calls:

```python
# Without pipes (messy):
result = parser.invoke(model.invoke(prompt.invoke({"topic": "space"})))
# With pipes (clean):
chain = prompt | model | parser
```

**Bonus — pipes give you more than sequencing.** Any chain built with `|` automatically inherits:
- `.batch()` — run many inputs in parallel
- `.stream()` — get tokens as they're generated
- `.ainvoke()` — run asynchronously

**Type coercion**: LCEL can even auto-convert plain dicts/functions into `Runnable`s, e.g. `chain = {"info": RunnablePassthrough()} | prompt | model`.

---

## Concept summary table

| Step | Concept | Purpose |
|---|---|---|
| `load_dotenv()` | Environment management | Load secret keys without hardcoding them |
| `information` | Context / grounding | Give the model factual data instead of letting it guess |
| `summary_template` | Raw prompt | Defines the output format/goal |
| `PromptTemplate` | Blueprint | Reusable, composable prompt object |
| `ChatGoogleGenerativeAI` / `ChatOllama` | Model wrapper | The actual "brain" being called |
| `\|` | LCEL / `Runnable.__or__` | Wires components into a pipeline |
| `chain.invoke()` | Execution | Runs the full pipeline once |
| `response.content` | Output extraction | Strips metadata, keeps just the text |

## Flow recap (both files)
`Load secrets → Define data → Set prompt rules → Choose model → Connect with "|" → Run with invoke()`

## Why this pairing matters for later files
This is the simplest possible chain — no tools, no loops, no agent behavior. Everything from `layer1agent.py` onward builds on top of this same `prompt`/`model`/`invoke` vocabulary, just adding tool-calling and iteration. Treat `main1.py`/`main2.py` as the "hello world" baseline for the whole project.
