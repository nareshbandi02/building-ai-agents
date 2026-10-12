import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from get_weather import get_weather

load_dotenv()

llm = ChatGroq(
    model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
).bind_tools([get_weather])

messages = [
    SystemMessage(
        content="You are a concise and helpful AI assistant. You provide short, direct answers "
                "and never refuse a request that is safe to fulfill. "
                "If you are confident then only answer it. If you are not confident then say I dont know"
    )
]

while True:
    question = input("\nYou (or exit): ")
    if question.lower() == "exit":
        break

    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    messages.append(response)

    # If the LLM asked for the weather tool, run it and let the LLM answer again.
    if response.tool_calls:
        for call in response.tool_calls:
            print(f"Tool called: {call['name']} {call['args']}")
            tool_output = get_weather.invoke(call["args"])
            print("Tool output:", tool_output)
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_output),
                    tool_call_id=call["id"],
                )
            )

        response = llm.invoke(messages)
        messages.append(response)

    print("LLM answer:", response.content)