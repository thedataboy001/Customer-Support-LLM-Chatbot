# Streamlit UI for your LangGraph + TrainingModel chatbot

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from src.model import TrainingModel 


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬", layout="centered")

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful customer service assistant.\n"
        "IMPORTANT RULES:\n"
        "- Do NOT use placeholders or template variables such as {{Order Number}}, <order_id>, or similar.\n"
        "- If you need an order number, ask plainly: 'Please share your order number.'\n"
        "- Use natural language only.\n"
        "- Be concise, polite, and ask one question at a time.\n"
        "- Do not claim to have looked up an order unless the user provided lookup results.\n"
    )
)

MODEL_PATH_DEFAULT = "llama3_model_v1/meta-llama/checkpoint-1500"


# -----------------------------
# LangGraph state
# -----------------------------
class AgentState(dict):
    # for simplicity in Streamlit, a dict works fine:
    # {"messages": List[BaseMessage]}
    pass


def messages_to_prompt(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"System: {m.content}")
        elif isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
        else:
            content = getattr(m, "content", str(m))
            lines.append(f"Message: {content}")
    return "\n".join(lines)


def build_app(llm: TrainingModel):
    def process(state: dict) -> dict:
        prompt = messages_to_prompt(state["messages"])

        # Add a short "generation anchor" to reinforce style right before generation
        anchored_prompt = (
            "Follow the rules strictly. Respond in natural language only. "
            "Do not use placeholders or template variables.\n\n"
            f"{prompt}\n\nAssistant:"
        )

        response_text = llm.generate(anchored_prompt)

        new_messages = state["messages"] + [AIMessage(content=response_text)]
        return {"messages": new_messages}

    graph = StateGraph(dict)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile()


# -----------------------------
# Cached model + graph
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_llm_and_graph(model_path: str):
    llm = TrainingModel(model_path)
    graph_app = build_app(llm)
    return llm, graph_app


# -----------------------------
# UI
# -----------------------------
st.title("Customer Support Chatbot")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=MODEL_PATH_DEFAULT)
    max_tokens = st.slider("Max new tokens", 32, 512, 120, 8)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05)

    if st.button("Reset chat"):
        st.session_state.pop("messages", None)
        st.session_state.pop("model_path_loaded", None)
        st.rerun()

# Load model/graph if needed (re-load if model_path changes)
if st.session_state.get("model_path_loaded") != model_path:
    _, graph_app = load_llm_and_graph(model_path)
    st.session_state["graph_app"] = graph_app
    st.session_state["model_path_loaded"] = model_path

# Initialize conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = [SYSTEM_PROMPT]

# Show chat history (skip showing system message)
for m in st.session_state["messages"]:
    if isinstance(m, SystemMessage):
        continue
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(m.content)

# Chat input
user_text = st.chat_input("Type your message…")
if user_text:
    st.session_state["messages"].append(HumanMessage(content=user_text))

    with st.chat_message("user"):
        st.markdown(user_text)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            graph_app = st.session_state["graph_app"]

            llm, _ = load_llm_and_graph(model_path)

            # Create a one-off graph using current generation settings
            def process_with_params(state: dict) -> dict:
                prompt = messages_to_prompt(state["messages"])
                anchored_prompt = (
                    "Follow the rules strictly. Respond in natural language only. "
                    "Do not use placeholders or template variables.\n\n"
                    f"{prompt}\n\nAssistant:"
                )
                response_text = llm.generate(
                    anchored_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                return {"messages": state["messages"] + [AIMessage(content=response_text)]}

            tmp_graph = StateGraph(dict)
            tmp_graph.add_node("process", process_with_params)
            tmp_graph.add_edge(START, "process")
            tmp_graph.add_edge("process", END)
            tmp_app = tmp_graph.compile()

            result = tmp_app.invoke({"messages": st.session_state["messages"]})
            st.session_state["messages"] = result["messages"]

            assistant_text = st.session_state["messages"][-1].content
            st.markdown(assistant_text)

# st.caption("Tip: type “exit” isn’t needed here — use Reset chat in the sidebar.")
