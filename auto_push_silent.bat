@echo off
setlocal enabledelayedexpansion
cd /d "C:\Users\Kabilan\Downloads\project"

git config --global user.name "kabilanthiyagarajan" >nul 2>&1
git config --global user.email "mukilankabilan40@gmail.com" >nul 2>&1
git config --global http.postBuffer 157286400 >nul 2>&1
git config --global http.lowSpeedLimit 0 >nul 2>&1
git config --global http.lowSpeedTime 999 >nul 2>&1
git config --global http.sslVerify false >nul 2>&1
git config --global credential.helper wincred >nul 2>&1

(
echo # Datasets
echo *.csv
echo *.tsv
echo *.json
echo *.jsonl
echo # Models
echo *.npy
echo *.npz
echo *.pkl
echo *.pickle
echo *.safetensors
echo *.bin
echo *.pt
echo *.pth
echo *.ckpt
echo *.h5
echo *.hdf5
echo # Folders
echo output/
echo outputs/
echo gpu_env/
echo venv/
echo env/
echo __pycache__/
echo *.pyc
echo *.log
echo .DS_Store
echo Thumbs.db
) > .gitignore 2>nul

if not exist ".git" (
    git init >nul 2>&1
    git branch -M main >nul 2>&1
)

git remote add origin https://github.com/kabilanthiyagarajan/Hybrid-Legal-Information-Retrieval-System-using-BM25-MiniLM-and-Cross-Encoder-Reranking.git 2>nul
git remote set-url origin https://github.com/kabilanthiyagarajan/Hybrid-Legal-Information-Retrieval-System-using-BM25-MiniLM-and-Cross-Encoder-Reranking.git 2>nul

git rm --cached *.csv *.npy *.npz *.pkl *.safetensors *.bin *.pt *.pth *.h5 2>nul
git rm -r --cached output/ outputs/ gpu_env/ venv/ 2>nul

git add .gitignore README.md push_to_github.ps1 auto_push_github.bat auto_push_silent.bat *.py *.html *.txt 2>nul

git commit -m "Auto-push: Hybrid Legal System - Code Only" >nul 2>&1
git push -u -f origin main >nul 2>&1
