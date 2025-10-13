# dsa4213-project

This repository contains all code, datasets, and results for our **DSA4213: Natural Language Processing for Data Science** group project.  
Our goal is to evaluate how **Retrieval-Augmented Generation (RAG)** improves **question answering (QA)** and **summarisation** of diabetes-related medical content, compared against **zero-shot** and **fine-tuned** baselines.

---

## Motivation

Access to accurate, understandable medical information is essential for public health. However, medical texts are often full of jargon and misinformation.  
This project investigates how **LLMs** (Large Language Models) with different setups—zero-shot, fine-tuned, and retrieval-augmented—can improve clarity, factuality, and accessibility in diabetes-related Q&A and summarisation tasks.

---

## Objectives

- Compare **Zero-shot**, **Fine-tuned**, and **RAG** approaches for diabetes Q&A and summarisation.  
- Evaluate the effect of **retrieval augmentation** (BM25, FAISS) and **prompting styles**.  
- Quantitatively assess model performance using **QA**, **summarisation**, and **readability metrics**.  
- Conduct **ablation studies** and **qualitative analysis** for factuality and readability.

---

## Dataset

**Primary Source:** [MedQuAD Dataset](https://catalog.data.gov/dataset/medquad-medical-question-answer-dataset)  
- ~47,000 QA pairs from 12 NIH/NLM medical websites  
- Filtered for **diabetes-related** questions and answers  
- Cleaned and preprocessed for training and retrieval  

| Split | Purpose | Percentage |
|--------|----------|------------|
| Train | Fine-tuning models | 70% |
| Validation | Model evaluation | 15% |
| Test | Final comparison | 15% |

---

## ⚙️ Methods & Models

### 1️⃣ Zero-shot Baseline
- Model: **FLAN-T5 (Generative)**
- Task: QA and summarisation without retrieval or fine-tuning  
- Evaluates LLM’s inherent understanding of diabetes-related queries  

### 2️⃣ Fine-tuned Models
- Models: **BioBERT (QA)**, **FLAN-T5 (Summarisation)**
- Dataset: Diabetes subset of MedQuAD  
- Evaluates how fine-tuning improves task-specific accuracy and readability  

### 3️⃣ Retrieval-Augmented Generation (RAG)
- Retriever: **BM25** (sparse) and **FAISS** (dense, BioBERT embeddings)  
- Generator: **FLAN-T5**  
- Input: Question + top-*k* retrieved passages (*k* = 1, 3, 5)  
- Evaluates how retrieval improves factual grounding and answer quality  

---

## Evaluation Metrics

| Task | Metrics |
|------|----------|
| QA | Exact Match (EM), F1 Score |
| Summarisation | ROUGE-1, ROUGE-2, ROUGE-L, BERTScore |
| Retrieval (RAG only) | Precision@k, Recall@k |
| Readability | Flesch-Kincaid Readability Score |
| Human Evaluation | Factuality and Clarity (5–10 samples) |

---

