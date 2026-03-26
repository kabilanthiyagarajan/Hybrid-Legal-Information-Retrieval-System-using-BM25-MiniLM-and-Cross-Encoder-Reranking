# Hidden Auto-Push - Runs silently in background
$ProgressPreference = "SilentlyContinue"
$ErrorActionPreference = "SilentlyContinue"

$USERNAME  = "kabilanthiyagarajan"
$EMAIL     = "mukilankabilan40@gmail.com"
$REPO      = "Hybrid-Legal-Information-Retrieval-System-using-BM25-MiniLM-and-Cross-Encoder-Reranking"
$FOLDER    = "C:\Users\Kabilan\Downloads\project"

if (!(Test-Path $FOLDER)) { exit }
Set-Location $FOLDER

git config --global user.name $USERNAME 2>$null
git config --global user.email $EMAIL 2>$null
git config --global http.postBuffer 157286400 2>$null
git config --global http.lowSpeedLimit 0 2>$null
git config --global http.lowSpeedTime 999 2>$null
git config --global http.sslVerify false 2>$null
git config --global credential.helper wincred 2>$null

$gitignore = @"
*.csv
*.tsv
*.json
*.jsonl
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
output/
outputs/
gpu_env/
venv/
env/
__pycache__/
*.pyc
*.log
.DS_Store
Thumbs.db
"@
$gitignore | Out-File -FilePath ".gitignore" -Encoding UTF8 2>$null

if (!(Test-Path ".git")) {
    git init 2>$null | Out-Null
    git branch -M main 2>$null
}

git remote remove origin 2>$null
git remote add origin "https://github.com/${USERNAME}/${REPO}.git" 2>$null

git rm --cached *.csv *.npy *.npz *.pkl *.safetensors *.bin *.pt *.pth *.h5 2>$null | Out-Null
git rm -r --cached output/ outputs/ 2>$null | Out-Null

git add .gitignore README.md push_to_github.ps1 auto_push_silent.bat auto_push_hidden.vbs auto_push_hidden.ps1 *.py *.html *.txt 2>$null | Out-Null

git commit -m "Auto-push: Hybrid Legal System" 2>$null | Out-Null
git push -u -f origin main 2>$null | Out-Null
