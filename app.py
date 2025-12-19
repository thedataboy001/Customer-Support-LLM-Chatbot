from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr


# ----------------------------------------------------------
# Load model the model path and Build the pipeline
# ----------------------------------------------------------

model_path = "llama3_model_v1/meta-llama/checkpoint-1500"

toks = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype = torch.bfloat16, 
                                             device_map = "auto"
                                             )
pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=toks)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------

# warmup
_ = pipe("Query: warmup\nResponse:", max_new_tokens=5, do_sample=False, return_full_text=False)


def response(message, history):

    prompt_parts = []
    for user, bot in history:
        prompt_parts.append(f"Query: {user}\nResponse: {bot}")
    prompt_parts.append(f"Query: {message}\nResponse:")

    prompt = "\n".join(prompt_parts)

    outputs = pipe(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=toks.eos_token_id,
    )

    generated_text = outputs[0]["generated_text"]
    bot_reply = generated_text[len(prompt):].strip()

    return bot_reply


# ---------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------


demo = gr.ChatInterface(fn=response, title = "Customer Support LLM Chatbot")

demo.launch(share=False)