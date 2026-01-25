#!/usr/bin/env python3
"""
TensorRT-LLM Translation Model Wrapper

This module provides a TensorRT-LLM based implementation of the translation model
that maintains API compatibility with the PyTorch-based TranslationModel class.

Key features:
- Drop-in replacement for TranslationModel in main.py
- 5-10x faster inference with INT8/INT4 quantization
- Identical generate() interface for seamless integration
- Supports greedy decoding (temperature=0) for deterministic translations

Usage:
    from trt_model import TensorRTTranslationModel

    llm = TensorRTTranslationModel(
        engine_dir="trt_engines/babel-9b-int8",
        tokenizer_dir="Babel-9B-Chat-Merged"
    )

    output = llm.generate(prompt="<|im_start|>user\\nHello<|im_end|>", temperature=0.0)
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp, SamplingConfig
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️  TensorRT-LLM not available. Please install: pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com")

from transformers import AutoTokenizer


class TensorRTTranslationModel:
    """
    TensorRT-LLM wrapper for translation model with PyTorch-compatible interface.

    This class loads a pre-compiled TensorRT engine and provides a generate() method
    that matches the signature of the PyTorch TranslationModel class in main.py.

    Attributes:
        engine_dir (str): Path to compiled TensorRT engine directory
        tokenizer_dir (str): Path to tokenizer files
        device (str): Device to use (always "cuda" for TensorRT)
        tokenizer: HuggingFace tokenizer
        runner: TensorRT-LLM model runner
    """

    def __init__(
        self,
        engine_dir: str,
        tokenizer_dir: str,
        device: str = "cuda"
    ):
        """
        Initialize TensorRT-LLM model.

        Args:
            engine_dir: Path to directory containing TensorRT engine files
            tokenizer_dir: Path to directory containing tokenizer files
            device: Device to use (only "cuda" supported)

        Raises:
            ImportError: If TensorRT-LLM is not installed
            FileNotFoundError: If engine_dir or tokenizer_dir doesn't exist
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT-LLM is not installed. "
                "Install with: pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
            )

        self.engine_dir = Path(engine_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.device = device

        # Validate paths
        if not self.engine_dir.exists():
            raise FileNotFoundError(f"TensorRT engine directory not found: {engine_dir}")

        if not self.tokenizer_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

        # Load tokenizer
        print(f"Loading tokenizer from: {tokenizer_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir),
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")

        # Load TensorRT engine
        print(f"Loading TensorRT engine from: {engine_dir}")
        try:
            self.runner = ModelRunnerCpp.from_dir(
                engine_dir=str(self.engine_dir),
                rank=0,  # Single GPU
                debug_mode=False  # Disable debug for production
            )
            print("✓ TensorRT engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load TensorRT engine: {e}")
            print("   Make sure the engine was compiled for your GPU architecture")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using TensorRT-LLM engine.

        This method signature matches TranslationModel.generate() in main.py
        to ensure drop-in compatibility.

        Args:
            prompt: Input text with chat template formatting
            max_tokens: Maximum new tokens to generate (default: 2048)
            temperature: Sampling temperature. Use 0.0 for greedy decoding (default: 0.0)
            stop: List of stop sequences (currently not implemented in TensorRT)

        Returns:
            Generated text as string (excluding the input prompt)

        Note:
            - temperature=0.0 enables greedy decoding (deterministic)
            - temperature>0.0 enables sampling with top-k/top-p
            - Stop sequences are handled by the translation pipeline, not here
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_length = input_ids.shape[1]

        # Prepare sampling configuration
        # For temperature=0, use greedy decoding (top_k=1)
        # For temperature>0, use nucleus sampling
        sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,

            # Temperature settings
            temperature=temperature if temperature > 0 else 0.0,

            # Sampling parameters
            top_k=1 if temperature == 0 else 50,  # Greedy vs sampling
            top_p=1.0 if temperature == 0 else 0.9,

            # Other generation settings
            repetition_penalty=1.1,  # Match PyTorch implementation
            beam_width=1,  # No beam search (greedy or sampling only)

            # Length penalties (disabled for now)
            length_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )

        # Run inference
        try:
            outputs = self.runner.generate(
                batch_input_ids=input_ids.tolist(),
                sampling_config=sampling_config,
                return_dict=True,
                output_sequence_lengths=True
            )
        except Exception as e:
            print(f"❌ TensorRT generation failed: {e}")
            raise

        # Extract generated tokens (exclude input prompt)
        # outputs['output_ids'] has shape: [batch_size, beam_width, sequence_length]
        output_ids = outputs['output_ids'][0][0]  # First batch, first beam

        # Remove input tokens to get only the generated part
        generated_ids = output_ids[input_length:]

        # Decode to text
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return generated_text.strip()

    def __repr__(self):
        return (
            f"TensorRTTranslationModel("
            f"engine={self.engine_dir.name}, "
            f"device={self.device})"
        )


# Alias for backward compatibility
TRTLLMModel = TensorRTTranslationModel


if __name__ == "__main__":
    print("TensorRT-LLM Translation Model Wrapper")
    print("=" * 70)
    print("\nThis module provides TensorRT-LLM inference for the translation API.")
    print("\nUsage:")
    print("  from trt_model import TensorRTTranslationModel")
    print("  llm = TensorRTTranslationModel(")
    print("      engine_dir='trt_engines/babel-9b-int8',")
    print("      tokenizer_dir='Babel-9B-Chat-Merged'")
    print("  )")
    print("  output = llm.generate(prompt='...')")
    print("\n" + "=" * 70)

    if not TENSORRT_AVAILABLE:
        print("\n⚠️  TensorRT-LLM is not installed!")
        print("Install with:")
        print("  pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com")
    else:
        print(f"\n✓ TensorRT-LLM version: {tensorrt_llm.__version__}")
