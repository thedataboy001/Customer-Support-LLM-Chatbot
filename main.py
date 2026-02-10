
import gradio as gr
from src.model import TrainingModel


# ---------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------

model = TrainingModel("llama3_model_v1/meta-llama/checkpoint-1500")

demo = gr.Interface(
    fn=model.generate,
    inputs=gr.Textbox(lines=2,
        label="Customer Query", placeholder="How can I help you today?"
    ),
    outputs=gr.Textbox(lines=20, label="Chatbot Response"),
    title="Customer Support LLM Chatbot"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
