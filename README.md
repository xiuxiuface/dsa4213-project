# DSA4213 Final Project  
## Assessing Model Adaptation Strategies for Biomedical Text Simplification: A Case Study on Diabetes

This repository contains the code, datasets, and results for our DSA4213: Natural Language Processing for Data Science project.  
We compare **Zero-Shot prompting**, **Fine-Tuning**, and **Retrieval-Augmented Generation (RAG)** for biomedical question answering (QA) and summarisation, using **diabetes-related content** from PubMedQA.

---

## Motivation  
Access to reliable medical information is essential for public health, yet biomedical literature is often difficult for non-experts to understand.  
This project investigates how different **LLM adaptation strategies** (zero-shot, fine-tuned, retrieval-augmented) can improve:

- clarity and readability  
- factual accuracy  
- biomedical grounding  

for diabetes-related QA and summarisation tasks.

---

## Objectives
- Compare **Zero-Shot**, **Fine-Tuned**, and **RAG** model setups.  
- Evaluate how **prompting**, **fine-tuning**, and **retrieval** affect performance.  
- Assess models using **QA accuracy**, **summarisation metrics**, and **readability scores**.  
- Perform **ablation studies** and **qualitative error analysis**.

---

## Dataset

### **Source**
- PubMedQA (HuggingFace Dataset)

### **Filtering & Processing**
- Extracted diabetes-related samples using **MeSH labels**.  
- Cleaned, deduplicated, and standardised biomedical text.  

### **Final Datasets**
- **QA Dataset** (`qa_dataset.xlsx`)  
  Biomedical questions, abstract snippets, short answers, Yes/No labels.  
- **Summarisation Dataset** (`summrization_dataset.xlsx`)  
  Full abstracts paired with simplified summaries.  
- **RAG Passage Dataset** (`rag_dataset.xlsx`)  
  Abstracts segmented into 3-sentence chunks for dense retrieval.

### **Data Split**
| Split | Purpose | Percentage |
|--------|----------|------------|
| Train | Model fine-tuning | **70%** |
| Validation | Hyperparameter tuning | **15%** |
| Test | Final comparison | **15%** |

---

## Methods & Notebooks

All experiments were conducted via Jupyter notebooks.

### **1. Data Preparation**  
- `PubMedQA_Preprocessing.ipynb`

### **2. Zero-Shot Prompting (FLAN-T5)**  
- `Zero_Shot_with_FLAN_T5.ipynb`

### **3. Fine-Tuning BioBERT for QA**  
- `finetuned_BioBERT.ipynb`

### **4. Fine-Tuning FLAN-T5 for Summarisation**  
- `finetuned_flant5.ipynb`

### **5. RAG for QA**  
- `rag-qa-final-new.ipynb`

### **6. RAG for Summarisation**  
- `rag-sum-full-new.ipynb`

### **7. Evaluation & Visualisation**  
- `Evaluation_Plots.ipynb`  
- `plot.py`

---

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| **QA** | Exact Match (EM), F1 Score |
| **Summarisation** | ROUGE-1, ROUGE-2, ROUGE-L, BERTScore |
| **Readability** | Flesch-Kincaid Readability Score |

---

## Summary  
This work benchmarks three major LLM adaptation strategies for biomedical tasks and highlights the trade-offs between **accuracy**, **readability**, and **evidence grounding**.  
All results, generated outputs, and plots are included in this repository.

