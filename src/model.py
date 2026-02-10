from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



# ----------------------------------------------------------
# Training Model
# ----------------------------------------------------------


@dataclass
class TrainingModel:
    model_path: str
    device_map: str = "auto"
    dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        if self.dtype is None:
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            torch_dtype = self.dtype,
            device_map = self.device_map,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype = self.dtype,
            device_map = self.device_map,
        )

        self.pipeline = pipeline(
            "text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
        )

    def generate(
            self,
            message: str,
            max_new_tokens: int = 120,
            temperature: float = 0.7,
            top_p: float = 0.9,
            repetition_penalty: float = 1.1,
    ) -> str:
        prompt = f"Query: {message}\nResponse:"

        outputs = self.pipeline(
            prompt,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            return_full_text = False,
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
        )

        response = outputs[0]["generated_text"].strip()

        if "Query:" in response:
            response = response.split("Query", 1)[0].strip()

        return response


# ---------------------------------------------------------------
# Testing 
# ---------------------------------------------------------------

if __name__ == "__main__":
    model = TrainingModel("llama3_model_v1/meta-llama/checkpoint-1500")
    print(model.generate("when should I be expecting my delivery"))
