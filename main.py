from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, List, Any, Mapping

import json
import json5
import re
import os
from pydantic import BaseModel, Field, ValidationError

# -------------------------------
# JSON REPAIR HELPERS
# -------------------------------
def clean_output(text: str) -> str:
    text = re.sub(r"^\s*Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?s)^Note:.*?\n", "", text)
    text = re.sub(r"(?s)^Disclaimer:.*?\n", "", text)
    text = re.sub(r"(?s)^.*?(\{)", r"\1", text)
    text = re.sub(r"(\}).*$", r"\1", text)
    return text.strip()

def repair_json(output: str) -> dict:
    # Find the first JSON object by tracking braces
    start_idx = output.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found")

    brace_count = 0
    end_idx = -1

    for i in range(start_idx, len(output)):
        if output[i] == '{':
            brace_count += 1
        elif output[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    # If incomplete, try to salvage it by closing the JSON
    if end_idx == -1:
        json_str = output[start_idx:]

        # Try to close any open string
        if json_str.count('"') % 2 == 1:
            json_str += '"'

        # Add closing brace
        json_str += '\n}'
    else:
        json_str = output[start_idx:end_idx]

    # Try parsing with standard json
    try:
        return json.loads(json_str)
    except Exception:
        pass

    # Try with json5
    try:
        return json5.loads(json_str)
    except Exception:
        pass

    # Try basic cleanup
    json_str = json_str.replace("\n", " ")
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Could not repair JSON: {e}")

class TranslationOutput(BaseModel):
    original_text: str = Field(...)
    translated_text: str = Field(...)

def validate_output(json_obj: dict) -> TranslationOutput:
    try:
        return TranslationOutput(**json_obj)
    except ValidationError:
        fixed = {
            "original_text": json_obj.get("original_text", ""),
            "translated_text": json_obj.get("translated_text", "")
        }
        return TranslationOutput(**fixed)

# -------------------------------
# MODEL CONFIGURATION
# -------------------------------
class TranslationModel:
    """Wrapper for HuggingFace transformers model"""
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0, stop: Optional[List[str]] = None) -> str:
        """Generate text using the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the input prompt)
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text.strip()

# -------------------------------
# MODEL LOADING WITH TRANSFORMERS
# -------------------------------
MODEL_NAME = "Babel-9B-Chat"

print("Loading model with transformers...")

# Load model
llm = TranslationModel(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

print("✓ Model loaded successfully!")

# -------------------------------
# PROMPTS (FIXED ESCAPING)
# -------------------------------
def build_prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    """Build the translation prompt"""
    system_prompt = """You are a professional translation engine. Follow these rules STRICTLY:

1. Translate EXACTLY what is written - do NOT add words, context, or interpretations
2. Provide LITERAL, word-for-word translation only
3. Do NOT embellish, explain, or add details not in the original
4. If uncertain, translate in humanlike way - never guess or infer meaning
5. Output ONLY valid JSON in a code block
6. Stop immediately after closing ```
7. Do not use any language other than the target language in the translation

Schema:
{
  "original_text": "<original>",
  "translated_text": "<Humanlike translation only - no additions>"
}

CRITICAL: The translated_text must contain ONLY the direct translation. No extra words."""

    few_shot_example = """Translate from Bangla to English.

IMPORTANT: Provide a HUMANLIKE, DIRECT translation. Do NOT add any words or context not present in the original text.

TEXT:
আমি ভাত খাই

Output ONLY this JSON:
```json
{
  "original_text": "আমি ভাত খাই",
  "translated_text": "I eat rice"
}
```"""

    user_prompt = f"""Translate the following text from {src_lang} to {tgt_lang}.

IMPORTANT: Provide a HUMANLIKE, DIRECT translation. Do NOT add any words or context not present in the original text.

TEXT:
{text}

Output ONLY this JSON:
```json
{{
  "original_text": "{text}",
  "translated_text": "<your humanlike translation here>"
}}
```"""

    # Combine into a single prompt
    full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{few_shot_example}<|im_end|>
<|im_start|>assistant
```json
{{
  "original_text": "আমি ভাত খাই",
  "translated_text": "I eat rice"
}}
```<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

    return full_prompt

# -------------------------------
# TRANSLATION FUNCTION
# -------------------------------
def translate(text: str, src_lang: str, tgt_lang: str) -> TranslationOutput:
    """
    Translate text from source language to target language.
    Uses strict prompting to minimize hallucination.
    """
    prompt = build_prompt(text, src_lang, tgt_lang)

    raw = llm.generate(
        prompt=prompt,
        max_tokens=2048,
        temperature=0.0,
        stop=["```\n", "```\n\n", "<|im_end|>", "\n\nTranslate", "}\n```"]
    )

    step1 = clean_output(raw)
    step2 = repair_json(step1)
    return validate_output(step2)

# -------------------------------
# DEMO
# -------------------------------
if __name__ == "__main__":
    test_cases = [
        ("The Bishop of Ramsbury was an episcopal title used by medieval English-Catholic diocesan bishops in the Anglo-Saxon English church. The title takes its name from the village of Ramsbury in Wiltshire, and was first used in the 10th and 11th centuries by the Anglo-Saxon Bishops of Ramsbury. In Saxon times, Ramsbury was an important location for the Church, and several early bishops became Archbishops of Canterbury.", "English", "Bangla")
    ]

    for text, src, tgt in test_cases:
        print(f"\n{'='*60}")
        print(f"Input: {text}")
        print(f"Direction: {src} → {tgt}")
        print('='*60)

        result = translate(text, src, tgt)
        print(f"Output: {result.translated_text}")
        print(f"JSON: {result.model_dump()}")
