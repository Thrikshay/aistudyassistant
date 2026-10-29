import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq


load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env")


# Override this with GROQ_MODEL in .env if your account uses another model.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI Study Assistant.

Your responsibilities:
- Explain concepts in simple language.
- Provide examples when useful.
- Answer follow-up questions using conversation history.
- Help students understand difficult topics.
- If you do not know an answer, say so.
""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm


# Conversation history is stored in memory and is cleared when the program exits.
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the existing history for a session, or create one."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


study_assistant = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def print_history(session_id: str) -> None:
    """Print the conversation stored for one session."""
    messages = get_session_history(session_id).messages
    if not messages:
        print("No conversation history yet.\n")
        return

    for message in messages:
        role = "You" if message.type == "human" else "Assistant"
        print(f"{role}: {message.content}")
        print()


def main() -> None:
    session_id = "student_1"

    print("Welcome to AI Study Assistant!")
    print("Type 'exit' to quit.")
    print("Type 'history' to see the conversation.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye! Happy studying!")
            break

        if user_input.lower() == "history":
            print_history(session_id)
            continue

        if not user_input:
            continue

        try:
            response = study_assistant.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"\nAssistant: {response.content}\n")
        except Exception as error:
            print(f"Error: {error}\n")


if __name__ == "__main__":
    main()
