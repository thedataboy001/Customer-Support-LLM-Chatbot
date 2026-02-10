import os
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from model import TrainingModel
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[BaseMessage]


# SYSTEM_PROMPT = SystemMessage(
#     content=(
#         "You are a helpful customer service assistant. "
#         "Be concise, ask clarifying questions when needed, and be polite. "
#         "If the customer didn't include an Order Number, don't include any Order Number"
#         "If you don't know something, say so."
#     )
# )

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful customer service assistant.\n"
        "IMPORTANT RULES:\n"
        "- Do NOT use placeholders or template variables such as {{Order Number}}, <order_id>, or similar.\n"
        "- If you need an order number, ask the user plainly, for example: "
        "'Please provide your order number.'\n"
        "- Use natural language only.\n"
        "- Be concise and polite."
    )
)

llm = TrainingModel("llama3_model_v1/meta-llama/checkpoint-1500")

@traceable
def messages_to_prompt(messages: List[BaseMessage]) -> str:
    """Convert LangChain messages into a plain-text chat transcript prompt."""
    lines: List[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"System: {m.content}")
        elif isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
        else:
            # Fallback for any other message type
            content = getattr(m, "content", str(m))
            lines.append(f"Message: {content}")
    return "\n".join(lines)

@traceable(run_type="llm")
def process(state: AgentState) -> AgentState:
    """
    Takes the current conversation state, generates the assistant response,
    and returns an updated state (no in-place mutation).
    """
    prompt = messages_to_prompt(state["messages"])
    response_text = llm.generate(prompt)

    new_messages = state["messages"] + [AIMessage(content=response_text)]
    print(f"\nAI: {response_text}")
    return {"messages": new_messages}


def build_app():
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile()

@traceable
def main():
    app = build_app()

    # Always pin the system prompt at the beginning
    conversation_history: List[BaseMessage] = [SYSTEM_PROMPT]

    while True:
        user_input = input("Enter: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        conversation_history.append(HumanMessage(content=user_input))

        try:
            result = app.invoke({"messages": conversation_history})
            conversation_history = result["messages"]
        except Exception as e:
            print(f"\n[error] {e}")


if __name__ == "__main__":
    main()
