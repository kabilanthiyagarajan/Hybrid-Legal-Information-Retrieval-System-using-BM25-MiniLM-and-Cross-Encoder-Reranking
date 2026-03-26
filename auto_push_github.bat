@echo off
color 0A
title GitHub Auto Push — CODE ONLY (No Dataset)
echo.
echo =====================================================
echo   AUTO PUSH — CODE ONLY (No CSV / NPY / Models)
echo =====================================================
echo.

:: ── Go to project folder ──────────────────────────────
cd /d "C:\Users\Kabilan\Downloads\project"
if errorlevel 1 (
    color 0C
    echo [ERROR] Folder not found: C:\Users\Kabilan\Downloads\project
    pause & exit
)
echo [OK] Found project folder

:: ── Increase Git buffer (fixes RPC/curl errors) ───────
git config --global http.postBuffer 157286400
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999
git config --global http.sslVerify false
echo [OK] Git buffer increased

:: ── Configure Git ─────────────────────────────────────
git config --global user.name  "kabilanthiyagarajan"
git config --global user.email "mukilankabilan40@gmail.com"
echo [OK] Git configured

:: ── Write .gitignore — blocks ALL large files ─────────
echo Writing .gitignore to block datasets and models...
(
echo # ── Datasets ──────────────────────────────────────
echo *.csv
echo *.tsv
echo *.json
echo *.jsonl
echo # ── Model / Embedding files ───────────────────────
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
echo # ── Large output folders ──────────────────────────
echo output/
echo outputs/
echo gpu_env/
echo venv/
echo env/
echo __pycache__/
echo *.pyc
echo # ── Logs ──────────────────────────────────────────
echo *.log
echo logs/
echo # ── OS files ──────────────────────────────────────
echo .DS_Store
echo Thumbs.db
) > .gitignore
echo [OK] .gitignore created

:: ── Init Git ──────────────────────────────────────────
if not exist ".git" (
    git init
    git branch -M main
    echo [OK] Git initialized
) else (
    echo [OK] Git already initialized
)

:: ── Connect remote ────────────────────────────────────
git remote add origin https://github.com/kabilanthiyagarajan/Hybrid-Legal-Information-Retrieval-System-using-BM25-MiniLM-and-Cross-Encoder-Reranking.git 2>nul
git remote set-url origin https://github.com/kabilanthiyagarajan/Hybrid-Legal-Information-Retrieval-System-using-BM25-MiniLM-and-Cross-Encoder-Reranking.git
echo [OK] Remote connected

:: ── REMOVE large files from Git cache if tracked ──────
echo.
echo Removing large files from Git tracking...
git rm --cached *.csv         2>nul
git rm --cached *.npy         2>nul
git rm --cached *.npz         2>nul
git rm --cached *.pkl         2>nul
git rm --cached *.safetensors 2>nul
git rm --cached *.bin         2>nul
git rm --cached *.pt          2>nul
git rm --cached *.pth         2>nul
git rm --cached *.h5          2>nul
git rm -r --cached output/    2>nul
git rm -r --cached outputs/   2>nul
git rm -r --cached gpu_env/   2>nul
git rm -r --cached venv/      2>nul
echo [OK] Large files removed from tracking

:: ── Stage ONLY code files ─────────────────────────────
git add .gitignore
git add *.py          2>nul
git add *.md          2>nul
git add *.txt         2>nul
git add *.html        2>nul
git add *.bat         2>nul
echo [OK] Code files staged

:: ── Show what will be pushed ──────────────────────────
echo.
echo ── FILES BEING PUSHED (should be small) ────────────
git status --short
echo ────────────────────────────────────────────────────
echo.

:: ── Commit ────────────────────────────────────────────
git commit -m "Final Year Project: Hybrid Swiss Legal Retrieval - BM25 + MiniLM + Cross-Encoder [code only]"
if errorlevel 1 (
    echo [INFO] Nothing new to commit - already up to date
)
echo [OK] Committed

:: ── Push ──────────────────────────────────────────────
echo.
echo =====================================================
echo  Pushing to GitHub...
echo  Enter your credentials when asked:
echo    Username = kabilanthiyagarajan
echo    Password = paste your ghp_... token
echo =====================================================
echo.
git push -u -f origin main

:: ── Result ────────────────────────────────────────────
echo.
echo =====================================================
if errorlevel 1 (
    color 0C
    echo   [FAILED] Push failed - see error above
    echo.
    echo   Most likely fix:
    echo   1. Token expired? Go to github.com Settings
    echo      - Developer settings - Personal access tokens
    echo      - Generate new token - tick repo - copy ghp_
    echo   2. Run this in PowerShell then try again:
    echo      git config --global http.sslVerify false
) else (
    color 0A
    echo   SUCCESS! Project is LIVE on GitHub!
    echo.
    echo   Visit: github.com/kabilanthiyagarajan
)
echo =====================================================
echo.
pause