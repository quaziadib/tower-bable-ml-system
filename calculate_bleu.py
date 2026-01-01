#!/usr/bin/env python3
"""
BLEU Score Calculator for Translation Performance

This script calculates BLEU scores for the translation API by:
1. Loading test data from performance-check-data/test.json
2. Sending text_data to the translation API
3. Comparing API output against validated_text (reference)
4. Computing aggregate BLEU scores and statistics

Usage:
    python calculate_bleu.py [--api-url URL] [--limit N] [--no-api]

    --api-url: API endpoint (default: http://localhost:9000/translate)
    --limit: Number of samples to test (default: all)
    --no-api: Calculate BLEU between existing translations only (for testing)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from sacrebleu import BLEU
from statistics import mean, stdev
import time


def load_test_data(filepath: str, limit: int = None) -> List[Dict]:
    """Load test data from JSON file."""
    print(f"Loading test data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    print(f"Loaded {len(data)} test samples")
    return data


def translate_via_api(text: str, source_lang: str, target_lang: str, api_url: str) -> Tuple[str, float]:
    """
    Translate text using the API.

    Returns:
        Tuple of (translated_text, time_taken_seconds)
    """
    start_time = time.time()

    try:
        response = requests.post(
            api_url,
            json={
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        elapsed = time.time() - start_time
        return result.get("translated_text", ""), elapsed

    except Exception as e:
        print(f"API Error: {e}")
        return "", -1


def calculate_bleu_scores(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate BLEU scores using sacrebleu.

    Args:
        hypotheses: List of translation outputs (from API)
        references: List of reference translations (validated_text)

    Returns:
        Dictionary with BLEU score and related metrics
    """
    bleu = BLEU()

    # sacrebleu expects references as list of lists (for multiple references per hypothesis)
    refs = [[ref] for ref in references]

    # Calculate corpus-level BLEU
    score = bleu.corpus_score(hypotheses, list(zip(*refs)))

    # Calculate sentence-level BLEU scores
    sentence_scores = []
    for hyp, ref in zip(hypotheses, references):
        sent_score = bleu.sentence_score(hyp, [ref])
        sentence_scores.append(sent_score.score)

    return {
        "corpus_bleu": score.score,
        "corpus_bleu_bp": score.bp,  # Brevity penalty
        "sentence_bleu_mean": mean(sentence_scores) if sentence_scores else 0,
        "sentence_bleu_std": stdev(sentence_scores) if len(sentence_scores) > 1 else 0,
        "sentence_bleu_min": min(sentence_scores) if sentence_scores else 0,
        "sentence_bleu_max": max(sentence_scores) if sentence_scores else 0,
        "sentence_scores": sentence_scores
    }


def run_evaluation(
    test_data: List[Dict],
    api_url: str = "http://localhost:9000/translate",
    use_api: bool = True
) -> Dict:
    """
    Run BLEU evaluation on test data.

    Args:
        test_data: List of test samples with text_data and validated_text
        api_url: Translation API endpoint
        use_api: If True, translate via API. If False, use dummy translations for testing

    Returns:
        Dictionary with evaluation results
    """
    hypotheses = []
    references = []
    translation_times = []
    errors = 0

    total = len(test_data)
    print(f"\n{'='*70}")
    print(f"Starting evaluation of {total} samples...")
    print(f"{'='*70}\n")

    for idx, sample in enumerate(test_data, 1):
        text = sample["text_data"]
        reference = sample["validated_text"]
        source_lang = sample.get("source_language_code", "bn")
        target_lang = sample.get("target_language_code", "en")

        if use_api:
            # Translate using API
            hypothesis, elapsed = translate_via_api(text, source_lang, target_lang, api_url)

            if hypothesis:
                hypotheses.append(hypothesis)
                references.append(reference)
                translation_times.append(elapsed)

                # Print progress
                if idx % 10 == 0 or idx == 1:
                    print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%) - "
                          f"Avg time: {mean(translation_times):.2f}s")
            else:
                errors += 1
                print(f"Error on sample {idx}")
        else:
            # For testing: just use reference as hypothesis (should give BLEU=100)
            hypotheses.append(reference)
            references.append(reference)

    print(f"\n{'='*70}")
    print(f"Translation complete: {len(hypotheses)}/{total} successful, {errors} errors")
    print(f"{'='*70}\n")

    # Calculate BLEU scores
    print("Calculating BLEU scores...")
    bleu_results = calculate_bleu_scores(hypotheses, references)

    # Compile results
    results = {
        "total_samples": total,
        "successful": len(hypotheses),
        "errors": errors,
        "bleu_scores": bleu_results,
        "translation_times": {
            "mean": mean(translation_times) if translation_times else 0,
            "std": stdev(translation_times) if len(translation_times) > 1 else 0,
            "min": min(translation_times) if translation_times else 0,
            "max": max(translation_times) if translation_times else 0,
            "total": sum(translation_times) if translation_times else 0
        }
    }

    return results


def print_results(results: Dict):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*70)
    print("BLEU SCORE EVALUATION RESULTS")
    print("="*70)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples:     {results['total_samples']}")
    print(f"  Successful:        {results['successful']}")
    print(f"  Errors:            {results['errors']}")
    print(f"  Success rate:      {results['successful']/results['total_samples']*100:.2f}%")

    bleu = results['bleu_scores']
    print(f"\n🎯 BLEU Scores:")
    print(f"  Corpus BLEU:       {bleu['corpus_bleu']:.2f}")
    print(f"  Brevity Penalty:   {bleu['corpus_bleu_bp']:.4f}")
    print(f"  Sentence BLEU (mean): {bleu['sentence_bleu_mean']:.2f}")
    print(f"  Sentence BLEU (std):  {bleu['sentence_bleu_std']:.2f}")
    print(f"  Sentence BLEU (min):  {bleu['sentence_bleu_min']:.2f}")
    print(f"  Sentence BLEU (max):  {bleu['sentence_bleu_max']:.2f}")

    if results['translation_times']['mean'] > 0:
        times = results['translation_times']
        print(f"\n⏱️  Translation Times:")
        print(f"  Mean time:         {times['mean']:.3f}s")
        print(f"  Std deviation:     {times['std']:.3f}s")
        print(f"  Min time:          {times['min']:.3f}s")
        print(f"  Max time:          {times['max']:.3f}s")
        print(f"  Total time:        {times['total']:.2f}s ({times['total']/60:.2f} min)")

    print("\n" + "="*70)

    # Interpretation guide
    print("\n📖 BLEU Score Interpretation:")
    print("  < 10:  Almost useless")
    print("  10-19: Hard to get the gist")
    print("  20-29: Clear gist, significant grammatical errors")
    print("  30-39: Understandable, some errors")
    print("  40-49: High quality, minor errors")
    print("  50-59: Very high quality, minimal errors")
    print("  > 60:  Often better than human translation")
    print("="*70 + "\n")


def save_detailed_results(results: Dict, output_file: str = "bleu_results.json"):
    """Save detailed results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Detailed results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BLEU scores for translation API"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:9000/translate",
        help="Translation API endpoint URL"
    )
    parser.add_argument(
        "--test-file",
        default="performance-check-data/test.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of samples to test"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip API calls (for testing script only)"
    )
    parser.add_argument(
        "--output",
        default="bleu_results.json",
        help="Output file for detailed results"
    )

    args = parser.parse_args()

    # Load test data
    test_data = load_test_data(args.test_file, args.limit)

    # Run evaluation
    results = run_evaluation(
        test_data,
        api_url=args.api_url,
        use_api=not args.no_api
    )

    # Print results
    print_results(results)

    # Save detailed results
    save_detailed_results(results, args.output)


if __name__ == "__main__":
    main()
