
from RAG_Summarization_Complete import (
    BM25Retriever,
    FAISSRetriever,
    HybridRetriever,
    SummarizationModel,
    RAGPipeline,
    RAGEvaluator,
    load_data,
    prepare_corpus
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
from typing import List, Dict, Tuple

# Retrieval libraries
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Evaluation
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import textstat
# ============================================================================
# QUICK TEST WITH SUBSET
# ============================================================================

def quick_test(sample_size: int = 50):
    """
    Quick test with small sample to verify everything works
    
    Args:
        sample_size: Number of samples to test (default: 50)
    """
    print("="*80)
    print(f"QUICK TEST MODE - Using {sample_size} samples")
    print("="*80)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_data("rag_dataset (3).xlsx")
    
    # 2. Sample subset
    print(f"\n[2/5] Sampling {sample_size} questions...")
    test_df = df.drop_duplicates(subset=["question_id"]).sample(
        n=sample_size, 
        random_state=42
    ).reset_index(drop=True)
    print(f"✓ Test set size: {len(test_df)}")
    
    # 3. Prepare corpus
    print("\n[3/5] Preparing corpus...")
    corpus, passage_ids = prepare_corpus(df)
    
    # 4. Initialize models
    print("\n[4/5] Initializing models...")
    
    # Retrievers
    print("  - BM25...")
    bm25 = BM25Retriever(corpus)
    
    print("  - FAISS...")
    faiss = FAISSRetriever(corpus)
    
    # Summarizer
    print("  - FLAN-T5...")
    summarizer = SummarizationModel("google/flan-t5-base")
    
    # 5. Test each retriever
    print("\n[5/5] Testing retrievers...")
    
    configs = [
        {"name": "BM25_k3", "retriever": bm25, "k": 3},
        {"name": "FAISS_k3", "retriever": faiss, "k": 3},
    ]
    
    evaluator = RAGEvaluator()
    results = []
    
    for config in configs:
        print(f"\n  Testing {config['name']}...")
        
        pipeline = RAGPipeline(
            retriever=config['retriever'],
            summarizer=summarizer,
            k=config['k']
        )
        
        predictions = []
        references = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {config['name']}"):
            pred = pipeline.run(row['question'])
            predictions.append(pred)
            references.append(row['answer'])
        
        metrics, fk_grades = evaluator.evaluate_batch(predictions, references)
        
        result = {
            "config": config['name'],
            **metrics
        }
        results.append(result)
        
        print(f"    ROUGE-1: {metrics['ROUGE-1']:.4f}")
        print(f"    ROUGE-2: {metrics['ROUGE-2']:.4f}")
        print(f"    BERTScore: {metrics['BERTScore_F1']:.4f}")
        print(f"    Avg FK: {metrics['Avg_FK_Grade']:.2f}")
    
    # Save test results
    
    results_df = pd.DataFrame(results)
    # results_df.to_csv("quick_test_results.csv", index=False)
    results_dict = results_df.to_dict(orient='records')
    with open("quick_test_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    # test_df['rag_summary_bm25'] = [pipeline.run(q) for q in test_df['question'][:10]]
    test_df.to_csv("test_sample_outputs.csv", index=False)
    
    print("\n" + "="*80)
    print("✓ QUICK TEST COMPLETED")
    print("="*80)
    print("\nResults:")
    print(results_df.to_string())

    
    return results_df

# ============================================================================
# INSPECT SPECIFIC EXAMPLES
# ============================================================================

def inspect_examples(n: int = 3):
    """
    Inspect a few examples in detail
    
    Args:
        n: Number of examples to inspect
    """
    print("\n" + "="*80)
    print(f"INSPECTING {n} EXAMPLES")
    print("="*80)
    
    # Load data
    df = load_data("rag_dataset (3).xlsx")
    corpus, _ = prepare_corpus(df)
    
    # Sample
    sample = df.drop_duplicates(subset=["question_id"]).sample(n=n, random_state=42)
    
    # Initialize
    bm25 = BM25Retriever(corpus)
    faiss = FAISSRetriever(corpus)
    summarizer = SummarizationModel("google/flan-t5-base")
    
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}/{n}")
        print(f"{'='*80}")
        
        print(f"\n Question:\n{row['question']}")
        print(f"\n True Answer:\n{row['answer']}")
        
        # BM25
        print(f"\n BM25 Retrieved Context (top-3):")
        bm25_ctx = bm25.retrieve(row['question'], k=3)
        for j, doc in enumerate(bm25_ctx, 1):
            print(f"  [{j}] {doc[:200]}...")
        
        bm25_summary = summarizer.generate(row['question'], bm25_ctx)
        print(f"\n BM25 RAG Summary:\n{bm25_summary}")
        print(f"   FK Grade: {textstat.flesch_kincaid_grade(bm25_summary):.2f}")
        
        # FAISS
        print(f"\n FAISS Retrieved Context (top-3):")
        faiss_ctx = faiss.retrieve(row['question'], k=3)
        for j, doc in enumerate(faiss_ctx, 1):
            print(f"  [{j}] {doc[:200]}...")
        
        faiss_summary = summarizer.generate(row['question'], faiss_ctx)
        print(f"\n FAISS RAG Summary:\n{faiss_summary}")
        print(f"   FK Grade: {textstat.flesch_kincaid_grade(faiss_summary):.2f}")

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        # Inspect mode: python test_subset_code.py inspect
        inspect_examples(n=5)
    else:
        # Quick test mode: python test_subset_code.py
        quick_test(sample_size=50)
    
    print("\n Done!")
