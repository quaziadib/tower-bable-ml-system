# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Translation API using HuggingFace Transformers with the Tower-Babel/Babel-9B-Chat model, featuring a FastAPI backend.

## Development Commands

```bash
# Run API server (basic)
python api.py

# Run with auto-reload for development
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Install dependencies
pip install -r requirements.txt

# Run standalone translation tests
python main.py
```

## Architecture

### Model Integration

The backend uses **HuggingFace Transformers** with PyTorch for inference. Key implementation details:

- **TranslationModel Wrapper** ([main.py:92-125](main.py#L92-L125)): Custom wrapper class for the HuggingFace model with generation methods
- **GPU Configuration**: FP16 precision with automatic device mapping (`device_map="auto"`)
- **Chat Template**: Uses Qwen's `<|im_start|>` and `<|im_end|>` tokens for proper prompt formatting

### Translation Quality Controls

The system uses multiple techniques to minimize LLM hallucinations:

1. **Greedy Decoding** (`temperature=0.0`): Ensures deterministic, literal translations
2. **Few-Shot Prompting**: Includes a Bangla→English example in the prompt ([main.py:162-175](main.py#L162-L175))
3. **Strict System Instructions**: Emphasizes "LITERAL, word-for-word translation" and forbids embellishments
4. **JSON Schema Enforcement**: Forces structured output with `original_text` and `translated_text` fields

### JSON Repair Pipeline

LLM outputs are unreliable, so [main.py:14-88](main.py#L14-L88) implements a multi-stage repair system:

1. **`clean_output()`**: Strips "Assistant:", "Note:", preamble text, and trailing garbage
2. **`repair_json()`**: Brace-matching to extract incomplete JSON, attempts closure with `}`
3. **Fallback Parsing**: Tries `json` → `json5` → regex cleanup before failing
4. **`validate_output()`**: Pydantic validation with automatic field fixing

When modifying prompts or output parsing, always test with edge cases (incomplete JSON, extra commentary).

### API Structure

- **[api.py](api.py)**: FastAPI server with CORS enabled
- **[main.py](main.py)**: Core translation logic, model loading, prompt engineering
- **Endpoints**:
  - `POST /translate`: Main translation endpoint (accepts `text`, `source_language`, `target_language`)
  - `GET /health`: Server health check
  - `GET /`: API information

## Model Configuration

- **Model**: `Tower-Babel/Babel-9B-Chat` (9B parameters)
- **Inference**: HuggingFace Transformers with FP16, automatic device mapping
- **Max Tokens**: 512 for translations
- **First Run**: Model auto-downloads (~18GB), requires GPU with 16GB+ VRAM

## Important Notes

- The model name in code (`Tower-Babel/Babel-9B-Chat`) differs from README description (`Qwen2.5-7B-Instruct`)
- Temperature is set to 0.0 for greedy decoding (deterministic translations)
- API runs on port 8000 by default
