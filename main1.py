from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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
    llm = ChatGoogleGenerativeAI(temperature = 0, model = "gemini-2.5-flash")

    chain = summary_prompt_template | llm

    response = chain.invoke(input= {"information":information})
    print(response.content)
    
if __name__ == "__main__":
    main()
