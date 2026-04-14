from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

load_dotenv()

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
    print(f"Searching for {query} ...")
    return tavily.search(query=query)



llm = ChatOpenAI(temperature=0, model="gpt-5.4-nano")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an AI engineer using langchain in Paris (France) on Linkedin and list their details")})
    print(result)


if __name__ == "__main__":
    main()
