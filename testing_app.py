from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr


# ----------------------------------------------------------
# Load model the model path and Build the pipeline
# ----------------------------------------------------------

model_path = "llama3_model_v1/meta-llama/checkpoint-500"

toks = AutoTokenizer.from_pretrained(model_path)

if toks.pad_token is None:
    toks.pad_token = toks.eos_token

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

def response(message, history):
    # Reconstruct history to match your training format:
    # Query: {instruction}\nResponse: {response}
    prompt_parts = []
    for entry in history:
        role = "Query" if entry["role"] == "user" else "Response"
        content = entry["content"]
        prompt_parts.append(f"{role}: {content}")
    
    # Append the current message
    prompt_parts.append(f"Query: {message}\nResponse:")
    
    # join with a newline to match your merge_example function
    prompt = "\n".join(prompt_parts)

    outputs = pipe(
        prompt,
        max_new_tokens=128,      # Increased to ensure answers aren't cut off
        do_sample=True,
        temperature=0.7,         # Better for natural customer support responses
        repetition_penalty=1.1,
        return_full_text=False,  # This makes 'generated_text' ONLY the new stuff
        clean_up_tokenization_spaces=True,
        eos_token_id=toks.eos_token_id
    )

    bot_reply = outputs[0]["generated_text"].strip()
    
    # Safety check: If the model hallucinates a new "Query:", cut it off
    if "Query:" in bot_reply:
        bot_reply = bot_reply.split("Query:")[0].strip()

    return bot_reply


# ---------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------

demo = gr.ChatInterface(fn=response, title = "Customer Support LLM Chatbot")

demo.launch(share=False)