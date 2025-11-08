"""
Complete RAG Summarization Pipeline with Ablation Studies
Includes: BM25, FAISS, Ablation Studies, Full Evaluation
"""

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
# 1. DATA LOADING
# ============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load the RAG dataset"""
    df = pd.read_excel(file_path)
    print(f"Loaded dataset: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

# ============================================================================
# 2. CORPUS PREPARATION
# ============================================================================

def prepare_corpus(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Extract unique passages and their IDs
    Returns: (corpus, passage_ids)
    """
    unique_passages = df.drop_duplicates(subset=["passage_id"])
    unique_passages = unique_passages[unique_passages['passage'].notna()]
    corpus = unique_passages["passage"].tolist()
    corpus = unique_passages["passage"].tolist()
    passage_ids = unique_passages["passage_id"].tolist()
    print(f"Unique passages: {len(corpus)}")
    return corpus, passage_ids

# ============================================================================
# 3. BM25 RETRIEVER
# ============================================================================

class BM25Retriever:
    def __init__(self, corpus: List[str]):
        print("Initializing BM25...")
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus
        print("✓ BM25 ready")
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [self.corpus[i] for i in top_k_indices]

# ============================================================================
# 4. FAISS DENSE RETRIEVER
# ============================================================================

class FAISSRetriever:
    def __init__(self, corpus: List[str], model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Initializing FAISS with {model_name}...")
        self.embed_model = SentenceTransformer(model_name)
        self.corpus = corpus
        
        # Create embeddings
        print("Creating passage embeddings...")
        self.passage_embeddings = self.embed_model.encode(
            corpus, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = self.passage_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.passage_embeddings)
        self.index.add(self.passage_embeddings)
        print(f"✓ FAISS index ready with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        # Encode query
        query_vec = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # Search
        scores, indices = self.index.search(query_vec, k)
        return [self.corpus[i] for i in indices[0]]

# ============================================================================
# 5. HYBRID RETRIEVER (Optional Enhancement)
# ============================================================================

class HybridRetriever:
    """Combines BM25 and FAISS with weighted scoring"""
    def __init__(self, corpus: List[str], alpha: float = 0.5):
        self.bm25_retriever = BM25Retriever(corpus)
        self.faiss_retriever = FAISSRetriever(corpus)
        self.alpha = alpha  # Weight for BM25 (1-alpha for FAISS)
        self.corpus = corpus
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        # Get more candidates from each
        k_candidates = k * 2
        bm25_docs = self.bm25_retriever.retrieve(query, k_candidates)
        faiss_docs = self.faiss_retriever.retrieve(query, k_candidates)
        
        # Simple fusion: combine and deduplicate
        combined = []
        seen = set()
        
        # Interleave results (giving priority to early results)
        for i in range(k_candidates):
            if i < len(bm25_docs) and bm25_docs[i] not in seen:
                combined.append(bm25_docs[i])
                seen.add(bm25_docs[i])
            if i < len(faiss_docs) and faiss_docs[i] not in seen:
                combined.append(faiss_docs[i])
                seen.add(faiss_docs[i])
            if len(combined) >= k:
                break
        
        return combined[:k]

# ============================================================================
# 6. SUMMARIZATION MODEL
# ============================================================================

class SummarizationModel:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"Loading summarization model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print(f"✓ Model loaded on {self.device}")
    
    def generate(self, question: str, context: List[str], max_length: int = 150) -> str:
        # Format prompt
        context_str = " ".join(context)
        prompt = f"Question: {question}\n\nContext: {context_str}\n\nSummarize the answer based on the context:"
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

# ============================================================================
# 7. EVALUATION FUNCTIONS
# ============================================================================

class RAGEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def evaluate_batch(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against references
        Returns: dict with ROUGE, BERTScore, and FK metrics
        """
        # ROUGE scores
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # BERTScore
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        # Flesch-Kincaid Grade
        fk_grades = [textstat.flesch_kincaid_grade(p) for p in predictions]
        
        # Aggregate metrics
        metrics = {
            "ROUGE-1": np.mean(rouge_scores['rouge1']),
            "ROUGE-2": np.mean(rouge_scores['rouge2']),
            "ROUGE-L": np.mean(rouge_scores['rougeL']),
            "BERTScore_F1": F1.mean().item(),
            "Avg_FK_Grade": np.mean(fk_grades)
        }
        
        return metrics, fk_grades

# ============================================================================
# 8. RAG PIPELINE
# ============================================================================

class RAGPipeline:
    def __init__(
        self, 
        retriever, 
        summarizer: SummarizationModel,
        k: int = 3
    ):
        self.retriever = retriever
        self.summarizer = summarizer
        self.k = k
    
    def run(self, question: str) -> str:
        # Retrieve relevant passages
        context = self.retriever.retrieve(question, k=self.k)
        
        # Generate summary
        summary = self.summarizer.generate(question, context)
        
        return summary

# ============================================================================
# 9. ABLATION STUDY
# ============================================================================

def run_ablation_study(
    df: pd.DataFrame,
    corpus: List[str],
    summarizer: SummarizationModel,
    test_size: int = 1000,  # Use subset for faster ablation
    output_dir: str = "results"
):
    """
    Run ablation studies comparing different retrievers and k values
    """
    print("\n" + "="*80)
    print("STARTING ABLATION STUDIES")
    print("="*80)
    
    # Sample test set
    test_df = df.drop_duplicates(subset=["question_id"]).sample(n=test_size, random_state=42)
    
    # Initialize retrievers
    print("\nInitializing retrievers...")
    bm25_retriever = BM25Retriever(corpus)
    faiss_retriever = FAISSRetriever(corpus)
    hybrid_retriever = HybridRetriever(corpus)
    
    # Ablation configurations
    configs = [
        {"name": "BM25_k3", "retriever": bm25_retriever, "k": 3},
        {"name": "BM25_k5", "retriever": bm25_retriever, "k": 5},
        {"name": "BM25_k10", "retriever": bm25_retriever, "k": 10},
        {"name": "FAISS_k3", "retriever": faiss_retriever, "k": 3},
        {"name": "FAISS_k5", "retriever": faiss_retriever, "k": 5},
        {"name": "FAISS_k10", "retriever": faiss_retriever, "k": 10},
        {"name": "Hybrid_k3", "retriever": hybrid_retriever, "k": 3},
        {"name": "Hybrid_k5", "retriever": hybrid_retriever, "k": 5},
    ]
    
    evaluator = RAGEvaluator()
    ablation_results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")
        
        # Create pipeline
        pipeline = RAGPipeline(
            retriever=config['retriever'],
            summarizer=summarizer,
            k=config['k']
        )
        
        # Generate predictions
        predictions = []
        references = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{config['name']}"):
            pred = pipeline.run(row['question'])
            predictions.append(pred)
            references.append(row['answer'])
        
        # Evaluate
        metrics, _ = evaluator.evaluate_batch(predictions, references)
        
        # Store results
        result = {
            "config_name": config['name'],
            "retriever_type": config['name'].split('_')[0],
            "k": config['k'],
            **metrics
        }
        ablation_results.append(result)
        
        print(f"Results: {metrics}")
    
    # Save ablation results
    os.makedirs(output_dir, exist_ok=True)
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(f"{output_dir}/ablation_study.csv", index=False)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print("\nResults saved to:", f"{output_dir}/ablation_study.csv")
    print("\nSummary:")
    print(ablation_df.to_string())
    
    return ablation_df

# ============================================================================
# 10. FULL EVALUATION ON ENTIRE DATASET
# ============================================================================

def run_full_evaluation(
    df: pd.DataFrame,
    corpus: List[str],
    summarizer: SummarizationModel,
    retriever_type: str = "FAISS",  # Best from ablation
    k: int = 5,
    output_dir: str = "results"
):
    """
    Run evaluation on the entire dataset with the best configuration
    """
    print("\n" + "="*80)
    print(f"FULL EVALUATION: {retriever_type} with k={k}")
    print("="*80)
    
    # Initialize retriever
    if retriever_type.upper() == "BM25":
        retriever = BM25Retriever(corpus)
    elif retriever_type.upper() == "FAISS":
        retriever = FAISSRetriever(corpus)
    elif retriever_type.upper() == "HYBRID":
        retriever = HybridRetriever(corpus)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    # Get unique questions
    unique_df = df.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
    
    # Create pipeline
    pipeline = RAGPipeline(retriever=retriever, summarizer=summarizer, k=k)
    
    # Generate predictions
    print("\nGenerating summaries for entire dataset...")
    predictions = []
    references = []
    
    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df)):
        pred = pipeline.run(row['question'])
        predictions.append(pred)
        references.append(row['answer'])
    
    # Add to dataframe
    unique_df['rag_summary'] = predictions
    
    # Evaluate
    evaluator = RAGEvaluator()
    metrics, fk_grades = evaluator.evaluate_batch(predictions, references)
    unique_df['FK_grade'] = fk_grades
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    unique_df.to_csv(f"{output_dir}/rag_full_outputs.csv", index=False)
    
    with open(f"{output_dir}/rag_full_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nOutputs saved to: {output_dir}/")
    
    return unique_df, metrics

# ============================================================================
# 11. EXTRACT REQUIRED PASSAGES
# ============================================================================

def extract_required_passages(
    full_df: pd.DataFrame,
    passage_ids: List[int],
    output_dir: str = "results"
):
    """
    Extract specific passages for reporting
    """
    print("\n" + "="*80)
    print("EXTRACTING REQUIRED PASSAGES")
    print("="*80)
    
    subset_df = full_df[full_df['passage_id'].isin(passage_ids)].copy()
    
    os.makedirs(output_dir, exist_ok=True)
    subset_df.to_csv(f"{output_dir}/rag_required_passages.csv", index=False)
    
    print(f"\nExtracted {len(subset_df)} passages")
    print(f"Saved to: {output_dir}/rag_required_passages.csv")
    print("\nSummary:")
    print(subset_df[['passage_id', 'question', 'rag_summary', 'FK_grade']].to_string())
    
    return subset_df

# ============================================================================
# 12. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print("RAG SUMMARIZATION PIPELINE")
    print("="*80)
    
    # Configuration
    DATA_PATH = "rag_dataset.xlsx"
    OUTPUT_DIR = "results"
    
    # 1. Load data
    print("\n[Step 1/6] Loading data...")
    df = load_data(DATA_PATH)
    
    # 2. Prepare corpus
    print("\n[Step 2/6] Preparing corpus...")
    corpus, passage_ids = prepare_corpus(df)
    
    # 3. Initialize summarization model
    print("\n[Step 3/6] Loading summarization model...")
    summarizer = SummarizationModel("google/flan-t5-base")
    
    # 4. Run ablation study
    print("\n[Step 4/6] Running ablation study...")
    ablation_results = run_ablation_study(
        df=df,
        corpus=corpus,
        summarizer=summarizer,
        test_size=1000,  # Adjust based on computational resources
        output_dir=OUTPUT_DIR
    )
    
    # 5. Full evaluation with best config
    print("\n[Step 5/6] Running full evaluation...")
    full_df, metrics = run_full_evaluation(
        df=df,
        corpus=corpus,
        summarizer=summarizer,
        retriever_type="FAISS",  # Change based on ablation results
        k=5,
        output_dir=OUTPUT_DIR
    )
    
    # 6. Extract required passages
    print("\n[Step 6/6] Extracting required passages...")
    required_passages = [16771, 12220, 29568]  # Your target passages
    subset_df = extract_required_passages(
        full_df=full_df,
        passage_ids=required_passages,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*80)
    print("✓ ALL TASKS COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_DIR}/ablation_study.csv")
    print(f"  2. {OUTPUT_DIR}/rag_full_outputs.csv")
    print(f"  3. {OUTPUT_DIR}/rag_full_metrics.json")
    print(f"  4. {OUTPUT_DIR}/rag_required_passages.csv")

if __name__ == "__main__":
    main()
