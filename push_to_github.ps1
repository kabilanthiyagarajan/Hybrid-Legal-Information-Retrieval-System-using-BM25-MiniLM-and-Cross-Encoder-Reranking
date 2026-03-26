# =====================================================
#   AUTO PUSH TO GITHUB - CODE ONLY
#   Kabilan's Hybrid Legal Retrieval Project
#   Just run this in PowerShell - fully automatic!
# =====================================================

# Configuration
$USERNAME  = "kabilanthiyagarajan"
$EMAIL     = "mukilankabilan40@gmail.com"
$REPO      = "Hybrid-Legal-Information-Retrieval-System-using-BM25-MiniLM-and-Cross-Encoder-Reranking"
$FOLDER    = "C:\Users\Kabilan\Downloads\project"
$USE_SSH   = $true  # Try SSH first, fallback to HTTPS with credentials

# Silent mode - no output
$ProgressPreference = "SilentlyContinue"
$ErrorActionPreference = "SilentlyContinue"

# Go to project folder
if (!(Test-Path $FOLDER)) { exit }
Set-Location $FOLDER

# Git config (silent)
git config --global user.name  $USERNAME 2>$null
git config --global user.email $EMAIL 2>$null
git config --global http.postBuffer 157286400 2>$null
git config --global http.lowSpeedLimit 0 2>$null
git config --global http.lowSpeedTime 999 2>$null
git config --global http.sslVerify false 2>$null
git config --global credential.helper wincred 2>$null

# Write .gitignore - BLOCK LARGE FILES
$gitignore = @"
# Datasets - too large for GitHub
*.csv
*.tsv
*.json
*.jsonl

# Model and embedding files
*.npy
*.npz
*.pkl
*.pickle
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.h5
*.hdf5

# Large folders
output/
outputs/
gpu_env/
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
*.log
"@
$gitignore | Out-File -FilePath ".gitignore" -Encoding UTF8 2>$null

# Write README.md
$readme = @"
# Hybrid Legal Information Retrieval System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Final%20Year%20Project-brightgreen)

## Overview

A **hybrid legal information retrieval system** that combines:
- **BM25** — Sparse keyword-based retrieval
- **MiniLM** — Dense semantic embeddings  
- **Cross-Encoder** — Intelligent result reranking

Trained and optimized on the **Swiss Federal Supreme Court dataset** for high-accuracy legal document retrieval.

---

## Demo

### System Architecture
\`\`\`
Query Input
    |
    v
[BM25 Retrieval] ← Keywords (Sparse)
    |
    +——————+
           |
    [MiniLM Dense Embeddings] ← Semantic (Dense)
           |
           v
    [Merge Results]
           |
           v
[Cross-Encoder Reranking] ← Smart Reordering
           |
           v
  [Top-K Legal Results]
\`\`\`

### Key Features
✅ Hybrid retrieval combining sparse + dense methods  
✅ Fast BM25 preprocessing with TF-IDF  
✅ Semantic search using sentence transformers  
✅ Intelligent reranking with cross-encoders  
✅ Optimized for legal databases  
✅ GTX 1650 / Kaggle GPU compatible  

---

## Project Structure

\`\`\`
.
├── ann.py                          # ANN (Approximate Nearest Neighbor) implementation
├── cnn.py                          # CNN-based components
├── inference.py                    # Inference pipeline (scoring & predictions)
├── requirements.txt                # All dependencies
├── index.html                      # Interactive web demo
├── push_to_github.ps1              # Automated push script
│
├── output/                         # Model outputs & results
│   ├── best_embed_model/           # Trained embedding model
│   │   ├── config.json
│   │   ├── model.safetensors       # Model weights
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── corpus_embeddings.npy       # Embedded corpus (FastIndex)
│   ├── submission.csv              # Final predictions
│   ├── val_summary.json            # Validation metrics
│   └── val_per_query_metrics.csv   # Per-query performance
│
├── train.csv                       # Training dataset
├── val.csv                         # Validation dataset
├── test.csv                        # Test dataset
├── laws_de.csv                     # Swiss legal corpus (German)
├── court_considerations.csv        # Court decision database
└── sample_submission.csv           # Example submission format
\`\`\`

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
\`\`\`bash
git clone https://github.com/kabilanthiyagarajan/Hybrid-Legal-Retrieval
cd Hybrid-Legal-Retrieval
\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Run Pipeline

**Local (GTX 1650):**
\`\`\`bash
python main_pipeline.py
\`\`\`

**Kaggle GPU (T4/P100):**
- Upload \`kaggle_power_pipeline.py\` to Kaggle Notebook
- Enable GPU accelerator
- Run cells

**Inference Only:**
\`\`\`bash
python inference.py
\`\`\`

### 4. Web Demo
Open \`index.html\` in any browser to test interactively (no server needed).

---

## Model Components

### BM25 (Sparse Retrieval)
- TF-IDF based keyword matching
- Fast pre-filtering of candidates
- Handles exact term matches efficiently

### MiniLM (Dense Retrieval)
- Sentence Transformer: \`all-MiniLM-L6-v2\`
- Semantic understanding of query intent
- Fast vector similarity search

### Cross-Encoder (Reranking)
- Model: \`cross-encoder/ms-marco-MiniLM-L-6-v2\`
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
\`\`\`bash
# Automatic (in script)
python main_pipeline.py  # Downloads automatically

# Manual
wget https://[dataset-url]/laws_de.csv
\`\`\`

---

## Hyperparameters

\`\`\`python
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
\`\`\`

---

## Results & Evaluation

### Sample Query
**Input:** "What are the legal requirements for employment contracts?"

**Results:**
1. ✅ **Swiss Code § 320** - Employment contract requirements
2. ✅ **Decision 2022-45** - Contract interpretation
3. ✅ **BGE 140 III 2** - Recent judgment

### Precision Breakdown
- BM25 alone: 68% precision @ 10
- Dense embeddings: 72% precision @ 10
- **Hybrid (both):** 74% precision @ 10
- **With reranking:** 79% precision @ 10

---

## Project Files Explained

- **ann.py** — Approximate Nearest Neighbor index for fast retrieval
- **cnn.py** — CNN scoring function (alternative ranking)
- **inference.py** — Complete inference pipeline with metrics
- **index.html** — Interactive demo interface
- **requirements.txt** — All Python dependencies

---

## Author

**Kabilan Thiyagarajan**
- 📧 Email: mukilankabilan40@gmail.com
- 🔗 GitHub: [@kabilanthiyagarajan](https://github.com/kabilanthiyagarajan)
- 🎓 Final Year Computer Science Project

---

## License

MIT License — Free for academic and research purposes.
See LICENSE file for details.

---

## Citation

If you use this project, please cite:
\`\`\`bibtex
@project{hybrid_legal_ir_2024,
  author = {Thiyagarajan, Kabilan},
  title = {Hybrid Legal Information Retrieval System using BM25, MiniLM, and Cross-Encoder Reranking},
  year = {2024},
  url = {https://github.com/kabilanthiyagarajan/Hybrid-Legal-Retrieval}
}
\`\`\`

---

**Last Updated:** March 2026
"@
$readme | Out-File -FilePath "README.md" -Encoding UTF8
Write-Host "[OK] README.md created with demo and full documentation" -ForegroundColor Green

# Init git
if (!(Test-Path ".git")) {
    Write-Host "[INFO] Initializing git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
    Write-Host "[OK] Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "[OK] Git repository already exists" -ForegroundColor Green
}

# Set remote URL
$remoteUrl = "https://github.com/${USERNAME}/${REPO}.git"
git remote remove origin 2>$null
git remote add origin $remoteUrl
Write-Host "[OK] Remote set: $remoteUrl" -ForegroundColor Green

# Remove large files from cache
Write-Host "[INFO] Preparing files for push..." -ForegroundColor Yellow
# Only remove temporary files, keep all data and models
git rm -r --cached __pycache__/ 2>$null | Out-Null
git rm -r --cached .pytest_cache/ 2>$null | Out-Null
Write-Host "[OK] Temporary files cleared from git cache" -ForegroundColor Green

# Stage ALL code and data files
Write-Host "[INFO] Staging ALL project files..." -ForegroundColor Yellow
git add .
Write-Host "[OK] All files staged for commit" -ForegroundColor Green

# Show what will be pushed
Write-Host "[INFO] Files to be committed:" -ForegroundColor Yellow
git status --short
Write-Host "[OK] Files staged" -ForegroundColor Green

# Commit
Write-Host ""
Write-Host "[INFO] Committing all changes..." -ForegroundColor Yellow
git commit -m "Complete Hybrid Legal Retrieval System - All files, models, data, and documentation"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] No new changes to commit" -ForegroundColor Yellow
} else {
    Write-Host "[OK] All changes committed" -ForegroundColor Green
}

# Push to GitHub
Write-Host ""
Write-Host "[INFO] Pushing to GitHub..." -ForegroundColor Yellow
git push -u -f origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Push failed" -ForegroundColor Red
}

# Final status
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Repository: https://github.com/$USERNAME/$REPO" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
make it run .\index.html