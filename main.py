import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

system_prompt = SystemMessage(
    content="You are a concise and helpful AI assistant. You provide short, direct answers and never refuse a request that is safe to fulfill. If you are confident then only answer it.If you are not confident then say I dont know"
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
)

while True:
    question = input("\n enter the Question you want to ask or press 'q' to quit: ").strip()

    if question.lower() == "q":
        print("\nGoodbye! Have a great day!")
        break
    response = llm.invoke([system_prompt, HumanMessage(content=question)])
    print(f"\nBot: {response.content}")
   