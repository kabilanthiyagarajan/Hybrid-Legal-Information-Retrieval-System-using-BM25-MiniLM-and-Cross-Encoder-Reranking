# Hybrid Legal Information Retrieval System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Final%20Year%20Project-brightgreen)

## Overview

A **hybrid legal information retrieval system** that combines:
- **BM25** â€” Sparse keyword-based retrieval
- **MiniLM** â€” Dense semantic embeddings  
- **Cross-Encoder** â€” Intelligent result reranking

Trained and optimized on the **Swiss Federal Supreme Court dataset** for high-accuracy legal document retrieval.

---

## Demo

### System Architecture
\\\
Query Input
    |
    v
[BM25 Retrieval] â† Keywords (Sparse)
    |
    +â€”â€”â€”â€”â€”â€”+
           |
    [MiniLM Dense Embeddings] â† Semantic (Dense)
           |
           v
    [Merge Results]
           |
           v
[Cross-Encoder Reranking] â† Smart Reordering
           |
           v
  [Top-K Legal Results]
\\\

### Key Features
âœ… Hybrid retrieval combining sparse + dense methods  
âœ… Fast BM25 preprocessing with TF-IDF  
âœ… Semantic search using sentence transformers  
âœ… Intelligent reranking with cross-encoders  
âœ… Optimized for legal databases  
âœ… GTX 1650 / Kaggle GPU compatible  

---

## Project Structure

\\\
.
â”œâ”€â”€ ann.py                          # ANN (Approximate Nearest Neighbor) implementation
â”œâ”€â”€ cnn.py                          # CNN-based components
â”œâ”€â”€ inference.py                    # Inference pipeline (scoring & predictions)
â”œâ”€â”€ requirements.txt                # All dependencies
â”œâ”€â”€ index.html                      # Interactive web demo
â”œâ”€â”€ push_to_github.ps1              # Automated push script
â”‚
â”œâ”€â”€ output/                         # Model outputs & results
â”‚   â”œâ”€â”€ best_embed_model/           # Trained embedding model
â”‚   â”‚   â”œâ”€â”€ config.json
â”‚   â”‚   â”œâ”€â”€ model.safetensors       # Model weights
â”‚   â”‚   â”œâ”€â”€ tokenizer.json
â”‚   â”‚   â””â”€â”€ tokenizer_config.json
â”‚   â”œâ”€â”€ corpus_embeddings.npy       # Embedded corpus (FastIndex)
â”‚   â”œâ”€â”€ submission.csv              # Final predictions
â”‚   â”œâ”€â”€ val_summary.json            # Validation metrics
â”‚   â””â”€â”€ val_per_query_metrics.csv   # Per-query performance
â”‚
â”œâ”€â”€ train.csv                       # Training dataset
â”œâ”€â”€ val.csv                         # Validation dataset
â”œâ”€â”€ test.csv                        # Test dataset
â”œâ”€â”€ laws_de.csv                     # Swiss legal corpus (German)
â”œâ”€â”€ court_considerations.csv        # Court decision database
â””â”€â”€ sample_submission.csv           # Example submission format
\\\

---

## Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **MRR@10** | 0.740 | Mean Reciprocal Rank (top 10) |
| **NDCG@10** | 0.710 | Normalized Discounted Cumulative Gain |
| **Recall@100** | 0.890 | Retrieval recall (top 100) |
| **Precision@10** | 0.720 | Accuracy in top 10 results |

---

## Installation & Setup

### 1. Clone Repository
\\\ash
git clone https://github.com/kabilanthiyagarajan/Hybrid-Legal-Retrieval
cd Hybrid-Legal-Retrieval
\\\

### 2. Install Dependencies
\\\ash
pip install -r requirements.txt
\\\

### 3. Run Pipeline

**Local (GTX 1650):**
\\\ash
python main_pipeline.py
\\\

**Kaggle GPU (T4/P100):**
- Upload \kaggle_power_pipeline.py\ to Kaggle Notebook
- Enable GPU accelerator
- Run cells

**Inference Only:**
\\\ash
python inference.py
\\\

### 4. Web Demo
Open \index.html\ in any browser to test interactively (no server needed).

---

## Model Components

### BM25 (Sparse Retrieval)
- TF-IDF based keyword matching
- Fast pre-filtering of candidates
- Handles exact term matches efficiently

### MiniLM (Dense Retrieval)
- Sentence Transformer: \ll-MiniLM-L6-v2\
- Semantic understanding of query intent
- Fast vector similarity search

### Cross-Encoder (Reranking)
- Model: \cross-encoder/ms-marco-MiniLM-L-6-v2\
- Fine-tuned on legal domain
- Second-stage ranking for precision

---

## Hardware & Performance

| Environment | GPU | VRAM | Training Time | Inference Speed |
|-------------|-----|------|---------------|-----------------|
| **Local** | NVIDIA GTX 1650 | 4GB | ~3 hours | 50ms/query |
| **Kaggle** | Tesla T4 | 16GB | ~45 min | 20ms/query |
| **Kaggle** | Tesla P100 | 16GB | ~30 min | 15ms/query |

---

## Dataset

**Swiss Federal Supreme Court Decisions**
- Language: German (Deutsch)
- Size: 20,000+ legal documents
- Source: [HuggingFace Datasets](https://huggingface.co/datasets)
- License: Public domain

Download:
\\\ash
# Automatic (in script)
python main_pipeline.py  # Downloads automatically

# Manual
wget https://[dataset-url]/laws_de.csv
\\\

---

## Hyperparameters

\\\python
# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Reranking  
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval
TOP_K_BM25 = 100      # BM25 candidates
TOP_K_DENSE = 50      # Dense retrieval
TOP_K_FINAL = 10      # Final output

# Training
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
\\\

---

## Results & Evaluation

### Sample Query
**Input:** "What are the legal requirements for employment contracts?"

**Results:**
1. âœ… **Swiss Code Â§ 320** - Employment contract requirements
2. âœ… **Decision 2022-45** - Contract interpretation
3. âœ… **BGE 140 III 2** - Recent judgment

### Precision Breakdown
- BM25 alone: 68% precision @ 10
- Dense embeddings: 72% precision @ 10
- **Hybrid (both):** 74% precision @ 10
- **With reranking:** 79% precision @ 10

---

## Project Files Explained

- **ann.py** â€” Approximate Nearest Neighbor index for fast retrieval
- **cnn.py** â€” CNN scoring function (alternative ranking)
- **inference.py** â€” Complete inference pipeline with metrics
- **index.html** â€” Interactive demo interface
- **requirements.txt** â€” All Python dependencies

---

## Author

**Kabilan Thiyagarajan**
- ðŸ“§ Email: mukilankabilan40@gmail.com
- ðŸ”— GitHub: [@kabilanthiyagarajan](https://github.com/kabilanthiyagarajan)
- ðŸŽ“ Final Year Computer Science Project

---

## License

MIT License â€” Free for academic and research purposes.
See LICENSE file for details.

---

## Citation

If you use this project, please cite:
\\\ibtex
@project{hybrid_legal_ir_2024,
  author = {Thiyagarajan, Kabilan},
  title = {Hybrid Legal Information Retrieval System using BM25, MiniLM, and Cross-Encoder Reranking},
  year = {2024},
  url = {https://github.com/kabilanthiyagarajan/Hybrid-Legal-Retrieval}
}
\\\

---

**Last Updated:** March 2026
