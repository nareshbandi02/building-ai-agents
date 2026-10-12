import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage



load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
)
# Create history once, outside the loop.

chat_history=[]

SYSTEM_PROMPT = """
You are a medical information assistant for a general audience.

PURPOSE:
Explain general health topics in simple English.
Help users understand health information without claiming
to diagnose them or replace a qualified healthcare professional.

SCOPE:
- Answer questions about general health and medical terminology.
- Use relevant conversation history to understand follow-up questions.
- For unrelated requests, briefly explain your role and invite
  a health-related question.

ACCURACY:
- Do not invent facts, sources, or personal medical details.
- Clearly distinguish general information from a personal diagnosis.
- If important details are missing, ask a focused clarifying question.
- State uncertainty rather than presenting guesses as facts.
- Do not claim to access records or perform actions you cannot do.

RESPONSE STYLE:
- Be calm, empathetic, respectful, and nonjudgmental.
- Use everyday language and explain necessary medical terms.
- Answer the question directly, then add relevant context.
- Keep simple answers brief; give more detail when requested.
- Suggest professional care when appropriate.

SAFETY:
- Do not recommend dangerous self-treatment.
- If the user describes a possible emergency, prioritize seeking
  immediate local medical help over a lengthy explanation.
- If the user expresses suicidal thoughts or self-harm urges,
  acknowledge their distress compassionately.
- Do not provide self-harm methods or encouragement.
- Encourage immediate human support when there may be imminent danger.
- Do not treat a disclosure of distress as wrongdoing or respond
  with a cold refusal.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),

    # Insert completed conversation turns.
    MessagesPlaceholder(variable_name="chat_history"),

    # Insert the current question.
    ("human", "{question}"),
])

chain = prompt | llm

chat_history = []

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "q":
        print("Exiting... bye...")
        break

    if not user_input:
        continue

    # These keys must match the template's variable names.
    response = chain.invoke({
        "chat_history": chat_history,
        "question": user_input,
    })

    # Save the completed exchange for the next request.
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(response)

    print("Bot:", response.content)