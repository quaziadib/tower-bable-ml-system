#!/usr/bin/env python3
"""
Merge LoRA Adapter into Base Model

This script merges the QLoRA fine-tuned adapter (final_model/) into the
base Babel-9B-Chat model, producing a standalone merged model ready for
TensorRT-LLM conversion.

Usage:
    python scripts/merge_lora.py

Output:
    Babel-9B-Chat-Merged/ directory containing the merged model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
from pathlib import Path

# Paths
BASE_DIR = Path("/home/adib/tower-bable-ml-system")
BASE_MODEL = str(BASE_DIR / "Babel-9B-Chat")
ADAPTER_PATH = str(BASE_DIR / "final_model")
OUTPUT_DIR = str(BASE_DIR / "Babel-9B-Chat-Merged")

def main():
    print("="*70)
    print("LoRA Adapter Merge Script")
    print("="*70)

    # Check inputs exist
    if not os.path.exists(BASE_MODEL):
        print(f"❌ Base model not found: {BASE_MODEL}")
        return False

    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Adapter not found: {ADAPTER_PATH}")
        return False

    print(f"\n📁 Base Model: {BASE_MODEL}")
    print(f"📁 LoRA Adapter: {ADAPTER_PATH}")
    print(f"📁 Output Directory: {OUTPUT_DIR}\n")

    # Load base model
    print("⏳ Loading base model...")
    print("   This will take ~30 seconds and use ~18GB GPU memory")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ Base model loaded")

    # Load LoRA adapter
    print("\n⏳ Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✓ LoRA adapter loaded")

    # Merge weights
    print("\n⏳ Merging LoRA weights into base model...")
    print("   This merges the 188MB adapter into the 17GB base model")
    merged_model = model.merge_and_unload()
    print("✓ Merge complete!")

    # Save merged model
    print(f"\n⏳ Saving merged model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merged_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="5GB"  # Split into 5GB shards
    )
    print("✓ Model saved")

    # Copy tokenizer
    print("\n⏳ Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✓ Tokenizer saved")

    # Verify output
    print("\n" + "="*70)
    print("✅ MERGE SUCCESSFUL!")
    print("="*70)
    print(f"\nMerged model location: {OUTPUT_DIR}")

    # List output files
    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        print(f"Output files ({len(files)} total):")
        for f in sorted(files)[:10]:  # Show first 10 files
            fpath = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024*1024)
                print(f"  - {f} ({size_mb:.1f} MB)")
        if len(files) > 10:
            print(f"  ... and {len(files)-10} more files")

    print("\n✓ Ready for TensorRT conversion!")
    print("  Next step: python scripts/convert_to_trtllm.py")
    print("="*70)

    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
