from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr


# ----------------------------------------------------------
# Load model the model path and Build the pipeline
# ----------------------------------------------------------

model_path = "llama3_model_v1/meta-llama/checkpoint-1500"

toks = AutoTokenizer.from_pretrained(model_path)

if toks.pad_token is None:
    toks.pad_token = toks.eos_token

model = AutoModelForCausalLM.from_pretrained( model_path, 
                                             dtype = torch.bfloat16, 
                                             device_map = "cuda"
                                             )
pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=toks)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------


def response(message):

    prompt = (f"Query: {message}\nResponse:")

    outputs = pipe(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
        eos_token_id=toks.eos_token_id,
    )

    bot_reply = outputs[0]["generated_text"].strip()
    
    # Safety check: If the model hallucinates a new "Query:", cut it off
    if "Query:" in bot_reply:
        bot_reply = bot_reply.split("Query:")[0].strip()

    return bot_reply



# ---------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------


demo = gr.Interface(
    fn=response,
    inputs=gr.Textbox(lines=2,
        label="Customer Query", placeholder="How can I help you today?"
    ),
    outputs=gr.Textbox(lines=20, label="Chatbot Response"),
    title="Customer Support LLM Chatbot"
)

demo.launch()