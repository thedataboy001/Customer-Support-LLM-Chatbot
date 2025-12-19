from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr


# ----------------------------------------------------------
# Load model the model path and Build the pipeline
# ----------------------------------------------------------

model_path = "llama3_model_v1/meta-llama/checkpoint-1500"

toks = AutoTokenizer.from_pretrained(model_path)

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
        eos_token_id=toks.eos_token_id,
    )

    generated_text = outputs[0]["generated_text"]
    response = generated_text[len(prompt):].strip()

    return response



# ---------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------


demo = gr.Interface(
    fn=response,
    inputs=gr.Textbox(
        lines=2, placeholder="Title"
    ),
    outputs=gr.Textbox()
)

demo.launch()