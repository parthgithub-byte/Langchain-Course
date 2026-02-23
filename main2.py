from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """
alman Salim Khan (born Abdul Rashid Salim Salman Khan,[a] 27 December 1965) is an Indian actor, film producer, and television personality who predominantly works in Hindi films. In a career spanning over three decades, his awards include two National Film Awards as a film producer, and two Filmfare Awards as an actor.[3] He has been cited in the media as one of the most popular and commercially successful actors of Indian cinema.[4][5] Forbes included him in listings of the highest-paid celebrities in the world, in 2015 and 2018.[6][7][8]

Khan began his acting career with a supporting role in Biwi Ho To Aisi (1988), followed by his breakthrough with a leading role in Sooraj Barjatya's romantic drama Maine Pyar Kiya (1989), for which he was awarded the Filmfare Award for Best Male Debut. He established himself with other commercially successful films, including Lawrence D'Souza's romantic drama Saajan (1991), Barjatya's family dramas Hum Aapke Hain Koun..! (1994) and Hum Saath-Saath Hain (1999), the action film Karan Arjun (1995) and the comedy Biwi No.1 (1999). This followed a period of decline in romantic comedy, musicals and tragedy drama in the 2000s.

Khan resurrected his screen image with the action film Wanted (2009), and achieved greater stardom in the following decade by starring in the top-grossing action films Dabangg (2010), Bodyguard (2011), Ek Tha Tiger (2012), Dabangg 2 (2012), Kick (2014), and Tiger Zinda Hai (2017), and the dramas Bajrangi Bhaijaan (2015) and Sultan (2016). This was followed by a series of poorly received films which failed critically and commercially, with the exception of Bharat (2019) and Tiger 3 (2023). Khan has starred in the annual highest-grossing Hindi films of 10 individual years, the highest for any actor.[9]

In addition to his acting career, Khan is a television presenter and promotes humanitarian causes through his charity, Being Human Foundation.[10] He has been hosting the reality show Bigg Boss since 2010.[11] Khan's off-screen life is marred by controversy and legal troubles. In 2015, he was convicted of culpable homicide for a negligent driving case in which he ran over five people with his car, killing one, but his conviction was set aside on appeal.[12][13][14][15] On 5 April 2018, Khan was convicted in a blackbuck poaching case and sentenced to five years imprisonment.[16][17] On 7 April 2018, he was out on bail while an appeal was ongoing.
"""

    summary_template = """
    Given the context {information} about a person, I want you to create:
    1. Short summary
    2. Two interesting facts about the person
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()


"""
Here is the step-by-step breakdown of every object and concept used:

1. load_dotenv()
Concept: Environment Management.
This function looks for a .env file in your project folder and loads the variables inside it into your system's memory.

Why it's there: LangChain and the Google integration automatically look for an environment variable named GOOGLE_API_KEY. By calling this at the very top, you ensure the code has "permission" to talk to Gemini without you having to paste your secret key directly into the script.

2. The information variable
Concept: The Context (Data).
This is just a standard Python string. In the AI world, we call this the Context.

Why it's there: LLMs have a "Knowledge Cutoff" (they don't know what happened yesterday). By providing this text, you are using a technique called Grounding. You are telling the AI: "Don't guess; only use this specific text to answer my request."

3. summary_template
Concept: The Raw Prompt.
This is a string containing a placeholder: {information}.

The Concept: This defines the "Goal." You are telling the AI exactly how to format the output (1. Summary, 2. Two facts).

4. PromptTemplate Object
Concept: The Blueprint/Formatter.

This object takes your raw string (summary_template) and turns it into a smart object.

input_variables=["information"]: This tells LangChain to expect a dictionary key named "information".

template=summary_template: This is the actual text structure.

Why use this instead of an f-string? As your apps get bigger, you might have 50 different templates. PromptTemplate objects can be saved, loaded, and shared across different parts of your app much more easily than plain Python strings.

5. ChatGoogleGenerativeAI Object
Concept: The Model Wrapper (The Engine).

This is the connection to Google's servers.

temperature=0: This controls "creativity." 0 means "be factual and deterministic." Since you are summarizing a biography, you don't want the AI to "hallucinate" or get creative with the facts.

model="gemini-2.5-flash": This specifies which version of Gemini to use. "Flash" is the fast, cost-efficient model.

6. The | (Pipe) Operator
Concept: LCEL (LangChain Expression Language).

This is the "Chain." It connects the Blueprint to the Engine.

How it works: It creates a pipeline. It says: "Take the output of the template (the formatted string) and shove it directly into the Gemini model."

7. chain.invoke()
Concept: Execution.
This is the "Start" button.

The Input: You pass a dictionary {"information": information}.

The Process: 1. The PromptTemplate replaces {information} with the actual text about Salman Khan.
2. The resulting long string is sent to Gemini.
3. Gemini processes it and sends back a response object.

8. response.content
Concept: Output Extraction.
When Gemini responds, it sends back a complex AIMessage object that includes:

The text (content).

The number of tokens used.

Why the model stopped (safety filters, length, etc.).

.content: You use this to ignore the metadata and just get the final summary and facts to print to your console.

Summary of the Flow
    Load Secrets (dotenv)
    Define Data (information)
    Set Rules (PromptTemplate)
    Choose Brains (ChatGoogleGenerativeAI)
    Connect them (|)
    Run it (invoke)

    

# Few more lines on the pipe operator in LCEL:
The pipe operator (|) is the secret sauce that makes LangChain feel like a modern, streamlined framework rather than a mess of nested functions. While it looks like a simple visual divider, it is actually a powerful Python feature being used under the hood.

1. How it works (The Technical "Magic")
In standard Python, the | symbol is the Bitwise OR operator (used for binary math). However, Python allows Operator Overloading.
LangChain uses a base class called Runnable. Every component you imported—PromptTemplate and ChatGoogleGenerativeAI—inherits from this class.

The Runnable class has a special "dunder" method called __or__. When Python sees object_a | object_b, it internally checks if object_a has this method and runs:

(Python)
# What you write:
chain = prompt | model

# What Python actually executes:
chain = prompt.__or__(model)

This creates a new object called a RunnableSequence. This sequence knows that when it is "invoked," it must first run the prompt and then pass that exact output into the model.

2. The "Unix" Philosophy
The design is inspired by Unix/Linux terminal pipes.

Unix: cat file.txt | grep "error" (Take text → find errors).

LangChain: prompt | model | parser (Format text → generate AI response → clean up result).

This "Data Flow" approach is much easier to read than the alternative, which would be nested function calls:

(Python)
# The "Messy" way (without pipes):
result = parser.invoke(model.invoke(prompt.invoke({"topic": "space"})))
With the pipe, the logic flows naturally from left to right, just like you read.

3. Automatic "Type Coercion"
The pipe operator is smart. If you try to pipe a regular Python dictionary or a function into a LangChain component, LCEL will often "coerce" (convert) it into a Runnable automatically.

(Python)
# You can even start a chain with a dictionary
chain = {"info": RunnablePassthrough()} | prompt | model
4. Hidden Features: Batching & Streaming
When you use the pipe operator to create a chain, you aren't just getting a simple sequence. The resulting chain object automatically inherits high-level powers:

.batch(): Run 10 prompts at once in parallel.

.stream(): Get the AI's answer word-by-word as it's generated.

.ainvoke(): Run the chain asynchronously (so your app doesn't freeze while waiting for Gemini).

Summary
The pipe operator is more than just a shortcut; it's a declarative way of coding. You are telling Python what you want the pipeline to look like, and LangChain handles the how (data passing, error handling, and parallelization) behind the scenes.
"""
