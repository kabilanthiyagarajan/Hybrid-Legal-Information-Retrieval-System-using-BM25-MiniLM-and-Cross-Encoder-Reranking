"""
=====================================================================
 Swiss Legal Information Retrieval — GTX 1650 Optimized Pipeline
 Final Year Project
=====================================================================
 FIXES:
   - Switched from multilingual-e5-large (560MB, needs 8GB VRAM)
     to paraphrase-multilingual-MiniLM-L12-v2 (118MB, fits in 4GB)
   - Gradient checkpointing enabled
   - Gradient accumulation (effective batch = 8)
   - torch.cuda.empty_cache() after every batch
   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   - batch_size=2, max_seq_len=64 for training
   - Corpus encoding with batch_size=16
   - num_workers=0 (Windows fix)
=====================================================================
"""

# ── Set memory allocator BEFORE importing torch ──────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─── Standard library ────────────────────────────────────────────
import re
import json
import time
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")

# ─── Third-party ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from rank_bm25 import BM25Okapi

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────
BASE   = Path(r"C:\Users\Kabilan\Downloads\project")
OUTPUT = BASE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATA = {
    "train"     : BASE / "train.csv",
    "val"       : BASE / "val.csv",
    "test"      : BASE / "test.csv",
    "laws"      : BASE / "laws_de.csv",
    "court"     : BASE / "court_considerations.csv",
    "sample_sub": BASE / "sample_submission.csv",
}

# ─── Device ──────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Running on : {DEVICE}")
if DEVICE.type == "cuda":
    log.info(f"GPU        : {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"VRAM       : {total_mem:.1f} GB")

# ─── Hyper-parameters  (tuned for GTX 1650 4 GB) ─────────────────
CFG = dict(
    # ── Model choices ────────────────────────────────────────────
    # paraphrase-multilingual-MiniLM-L12-v2:
    #   • 118 MB on disk   • ~400 MB VRAM at inference
    #   • 12-layer MiniLM, 384-dim embeddings
    #   • Trained on 50+ languages — handles EN query + DE citations
    embed_model  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2",

    # ── Retrieval ─────────────────────────────────────────────────
    bm25_top_k   = 100,
    dense_top_k  = 100,
    rerank_top_k = 20,
    final_top_k  = 10,

    # ── Training (GTX 1650 safe values) ───────────────────────────
    train_batch      = 2,    # physical batch per GPU step
    grad_accum_steps = 4,    # effective batch = 2 * 4 = 8
    max_seq_len      = 64,   # shorter = less memory during training
    infer_seq_len    = 128,  # slightly longer ok at inference (no grads)
    infer_batch      = 16,   # corpus encoding batch size
    epochs           = 3,
    lr               = 2e-5,
    warmup_ratio     = 0.1,
    weight_decay     = 0.01,
    fp16             = torch.cuda.is_available(),
    num_workers      = 0,    # MUST be 0 on Windows
)


# ══════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_data():
    log.info("Loading datasets ...")
    dfs = {}
    for key, path in DATA.items():
        if path.exists():
            dfs[key] = pd.read_csv(path)
            log.info(f"  {key:12s}: {len(dfs[key]):,} rows  |  cols: {list(dfs[key].columns)}")
        else:
            log.warning(f"  {key} NOT FOUND at {path}")
    return dfs


# ══════════════════════════════════════════════════════════════════
# 2.  CORPUS BUILDING
# ══════════════════════════════════════════════════════════════════

def build_corpus(dfs):
    citations, texts = [], []

    # ── Federal laws ─────────────────────────────────────────────
    if "laws" in dfs:
        df       = dfs["laws"]
        text_col = next((c for c in ["text", "article_text", "de_text", "content"] if c in df.columns), None)
        cit_col  = next((c for c in ["citation", "id", "article_id"] if c in df.columns), None)
        if text_col and cit_col:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Laws corpus"):
                cit  = str(row[cit_col]).strip()
                body = str(row[text_col]).strip()[:300]   # truncate very long texts
                citations.append(cit)
                texts.append(f"{cit} {body}")
        else:
            cols = df.columns.tolist()
            log.warning("laws_de.csv: unexpected columns — using first two")
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Laws fallback"):
                citations.append(str(row[cols[0]]).strip())
                texts.append(" ".join(str(v) for v in row.values if pd.notna(v))[:300])

    # ── Court decisions ───────────────────────────────────────────
    if "court" in dfs:
        df       = dfs["court"]
        text_col = next((c for c in ["text", "consideration", "de_text", "content", "body"] if c in df.columns), None)
        cit_col  = next((c for c in ["citation", "id", "decision_id", "bge"] if c in df.columns), None)
        if text_col and cit_col:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Court corpus"):
                cit  = str(row[cit_col]).strip()
                body = str(row[text_col]).strip()[:300]
                citations.append(cit)
                texts.append(f"{cit} {body}")
        else:
            cols = df.columns.tolist()
            log.warning("court_considerations.csv: unexpected columns — using first two")
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Court fallback"):
                citations.append(str(row[cols[0]]).strip())
                texts.append(" ".join(str(v) for v in row.values if pd.notna(v))[:300])

    log.info(f"Corpus size: {len(citations):,} documents")
    return citations, texts


# ══════════════════════════════════════════════════════════════════
# 3.  BM25 INDEX
# ══════════════════════════════════════════════════════════════════

def tokenize_for_bm25(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_bm25(texts):
    log.info("Building BM25 index ...")
    tokenized = [tokenize_for_bm25(t) for t in tqdm(texts, desc="BM25 tokenize")]
    return BM25Okapi(tokenized)


# ══════════════════════════════════════════════════════════════════
# 4.  EMBEDDING MODEL  (MiniLM — fits GTX 1650)
# ══════════════════════════════════════════════════════════════════

class EmbeddingModel:
    def __init__(self, model_name):
        log.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=None, seq_len=None):
        """Encode a list of texts → L2-normalised embeddings (numpy)."""
        if batch_size is None:
            batch_size = CFG["infer_batch"]
        if seq_len is None:
            seq_len = CFG["infer_seq_len"]

        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i: i + batch_size]
            enc   = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            ).to(DEVICE)
            out  = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb  = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
            # Free GPU memory after each batch
            del enc, out, mask, emb
            torch.cuda.empty_cache()

        return np.vstack(all_embs)

    def encode_queries(self, queries, batch_size=32):
        return self.encode(queries, batch_size=batch_size, seq_len=CFG["infer_seq_len"])


# ══════════════════════════════════════════════════════════════════
# 5.  CROSS-ENCODER RERANKER
# ══════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    def __init__(self, model_name):
        log.info(f"Loading cross-encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def score(self, query, passages, batch_size=16):
        scores = []
        pairs  = [[query, p] for p in passages]
        for i in range(0, len(pairs), batch_size):
            batch  = pairs[i: i + batch_size]
            enc    = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=CFG["infer_seq_len"],
                return_tensors="pt",
            ).to(DEVICE)
            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(logits.cpu().float().numpy().tolist())
            del enc, logits
            torch.cuda.empty_cache()
        return np.array(scores)


# ══════════════════════════════════════════════════════════════════
# 6.  DATASET
# ══════════════════════════════════════════════════════════════════

class LegalPairDataset(Dataset):
    def __init__(self, df, citations, texts, tokenizer, max_len):
        self.samples  = []
        cit2idx       = {c: i for i, c in enumerate(citations)}
        query_col     = next(c for c in ["query", "question", "text"] if c in df.columns)
        gold_col      = next(c for c in ["gold_citations", "citations", "answer"] if c in df.columns)
        for _, row in df.iterrows():
            q = str(row[query_col])
            for g in str(row[gold_col]).split(";"):
                g = g.strip()
                if g in cit2idx:
                    self.samples.append((q, texts[cit2idx[g]]))
        self.tokenizer = tokenizer
        self.max_len   = max_len
        log.info(f"  Dataset: {len(self.samples):,} positive pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, p  = self.samples[idx]
        enc_q = self.tokenizer(q, truncation=True, max_length=self.max_len,
                               padding="max_length", return_tensors="pt")
        enc_p = self.tokenizer(p, truncation=True, max_length=self.max_len,
                               padding="max_length", return_tensors="pt")
        return {
            "q_input_ids"     : enc_q["input_ids"].squeeze(),
            "q_attention_mask": enc_q["attention_mask"].squeeze(),
            "p_input_ids"     : enc_p["input_ids"].squeeze(),
            "p_attention_mask": enc_p["attention_mask"].squeeze(),
        }


# ══════════════════════════════════════════════════════════════════
# 7.  CONTRASTIVE LOSS + MEAN POOL
# ══════════════════════════════════════════════════════════════════

def mean_pool(out, mask):
    mask = mask.unsqueeze(-1).float()
    return (out.last_hidden_state * mask).sum(1) / mask.sum(1)


def contrastive_loss(q_emb, p_emb, temp=0.05):
    q_emb  = F.normalize(q_emb, dim=-1)
    p_emb  = F.normalize(p_emb, dim=-1)
    sims   = q_emb @ p_emb.T / temp
    labels = torch.arange(len(q_emb)).to(q_emb.device)
    return F.cross_entropy(sims, labels)


# ══════════════════════════════════════════════════════════════════
# 8.  FINE-TUNING  (gradient accumulation + empty_cache)
# ══════════════════════════════════════════════════════════════════

def fine_tune_embedder(embed_model, train_df, val_df, citations, texts):
    log.info("Fine-tuning bi-encoder ...")
    log.info(f"  Physical batch : {CFG['train_batch']}")
    log.info(f"  Grad accum     : {CFG['grad_accum_steps']}")
    log.info(f"  Effective batch: {CFG['train_batch'] * CFG['grad_accum_steps']}")

    tokenizer = embed_model.tokenizer
    model     = embed_model.model

    # Enable gradient checkpointing to cut activation memory ~50%
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log.info("  Gradient checkpointing: ON")

    model.train()

    train_ds = LegalPairDataset(train_df, citations, texts, tokenizer, CFG["max_seq_len"])
    val_ds   = LegalPairDataset(val_df,   citations, texts, tokenizer, CFG["max_seq_len"])

    train_dl = DataLoader(train_ds, batch_size=CFG["train_batch"], shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["train_batch"], shuffle=False, num_workers=0, pin_memory=True)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler       = torch.cuda.amp.GradScaler(enabled=CFG["fp16"])
    total_steps  = (len(train_dl) // CFG["grad_accum_steps"]) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, CFG["epochs"] + 1):
        # ── Train ────────────────────────────────────────────────
        model.train()
        epoch_loss    = 0.0
        accum_count   = 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_dl), total=len(train_dl),
                    desc=f"Epoch {epoch}/{CFG['epochs']} [train]")
        for step, batch in pbar:
            with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
                q_out = model(input_ids      = batch["q_input_ids"].to(DEVICE),
                              attention_mask = batch["q_attention_mask"].to(DEVICE))
                p_out = model(input_ids      = batch["p_input_ids"].to(DEVICE),
                              attention_mask = batch["p_attention_mask"].to(DEVICE))
                q_emb = mean_pool(q_out, batch["q_attention_mask"].to(DEVICE))
                p_emb = mean_pool(p_out, batch["p_attention_mask"].to(DEVICE))
                loss  = contrastive_loss(q_emb, p_emb) / CFG["grad_accum_steps"]

            scaler.scale(loss).backward()
            epoch_loss  += loss.item() * CFG["grad_accum_steps"]
            accum_count += 1

            # Clear GPU tensors immediately
            del q_out, p_out, q_emb, p_emb, loss
            torch.cuda.empty_cache()

            if accum_count == CFG["grad_accum_steps"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

            pbar.set_postfix({"loss": f"{epoch_loss/(step+1):.4f}",
                              "mem": f"{torch.cuda.memory_allocated()/1e9:.1f}GB"})

        # flush leftover accumulation
        if accum_count > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train = epoch_loss / len(train_dl)

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch}/{CFG['epochs']} [val]"):
                with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
                    q_out = model(input_ids      = batch["q_input_ids"].to(DEVICE),
                                  attention_mask = batch["q_attention_mask"].to(DEVICE))
                    p_out = model(input_ids      = batch["p_input_ids"].to(DEVICE),
                                  attention_mask = batch["p_attention_mask"].to(DEVICE))
                    q_emb = mean_pool(q_out, batch["q_attention_mask"].to(DEVICE))
                    p_emb = mean_pool(p_out, batch["p_attention_mask"].to(DEVICE))
                    loss  = contrastive_loss(q_emb, p_emb)
                val_loss += loss.item()
                del q_out, p_out, q_emb, p_emb, loss
                torch.cuda.empty_cache()

        avg_val = val_loss / max(len(val_dl), 1)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        log.info(f"Epoch {epoch}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            save_dir = str(OUTPUT / "best_embed_model")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            log.info(f"  BEST model saved  val_loss={best_val:.4f}  -> {save_dir}")

        # Full memory flush between epochs
        torch.cuda.empty_cache()

    model.eval()
    return history


# ══════════════════════════════════════════════════════════════════
# 9.  RETRIEVAL PIPELINE
# ══════════════════════════════════════════════════════════════════

class LegalRetrievalPipeline:
    def __init__(self, citations, corpus_texts, bm25, embed_model, corpus_embs, reranker):
        self.citations    = citations
        self.corpus_texts = corpus_texts
        self.bm25         = bm25
        self.embed        = embed_model
        self.corpus_embs  = corpus_embs
        self.reranker     = reranker

    def _bm25_retrieve(self, query, top_k):
        scores = np.array(self.bm25.get_scores(tokenize_for_bm25(query)))
        return np.argsort(scores)[::-1][:top_k]

    def _dense_retrieve(self, query_emb, top_k):
        sims = (self.corpus_embs @ query_emb.T).squeeze()
        return np.argsort(sims)[::-1][:top_k]

    def retrieve(self, query, final_top_k=None):
        if final_top_k is None:
            final_top_k = CFG["final_top_k"]
        bm25_ids  = self._bm25_retrieve(query, CFG["bm25_top_k"])
        q_emb     = self.embed.encode_queries([query], batch_size=1)
        dense_ids = self._dense_retrieve(q_emb, CFG["dense_top_k"])
        cand_ids  = list(dict.fromkeys(np.concatenate([bm25_ids, dense_ids])))[:CFG["rerank_top_k"]]
        passages  = [self.corpus_texts[i] for i in cand_ids]
        rr_scores = self.reranker.score(query, passages)
        ranked    = sorted(zip(cand_ids, rr_scores), key=lambda x: x[1], reverse=True)
        return [self.citations[idx] for idx, _ in ranked[:final_top_k]]

    def batch_retrieve(self, queries, final_top_k=None):
        results = []
        for q in tqdm(queries, desc="Retrieving"):
            results.append(self.retrieve(q, final_top_k))
        return results


# ══════════════════════════════════════════════════════════════════
# 10.  EVALUATION
# ══════════════════════════════════════════════════════════════════

def citation_f1(pred, gold):
    pred_set, gold_set = set(pred), set(gold)
    if not gold_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 0.0, 0.0)
    tp = len(pred_set & gold_set)
    p  = tp / len(pred_set) if pred_set else 0.0
    r  = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


def evaluate_pipeline(pipeline, df, split_name="val"):
    query_col = next(c for c in ["query", "question", "text"] if c in df.columns)
    gold_col  = next(c for c in ["gold_citations", "citations", "answer"] if c in df.columns)
    queries   = df[query_col].tolist()
    golds     = [[g.strip() for g in str(row).split(";")] for row in df[gold_col].tolist()]
    preds     = pipeline.batch_retrieve(queries)

    f1s, ps, rs, per_query = [], [], [], []
    for i, (pred, gold) in enumerate(zip(preds, golds)):
        f1, p, r = citation_f1(pred, gold)
        f1s.append(f1); ps.append(p); rs.append(r)
        per_query.append({
            "query_id" : df.iloc[i].get("query_id", i),
            "query"    : queries[i],
            "gold"     : ";".join(gold),
            "predicted": ";".join(pred),
            "f1"       : round(f1, 4),
            "precision": round(p,  4),
            "recall"   : round(r,  4),
        })

    macro_f1 = float(np.mean(f1s))
    macro_p  = float(np.mean(ps))
    macro_r  = float(np.mean(rs))
    summary  = {
        "split"          : split_name,
        "num_queries"    : len(queries),
        "macro_f1"       : round(macro_f1, 4),
        "macro_precision": round(macro_p,  4),
        "macro_recall"   : round(macro_r,  4),
    }

    log.info(f"\n{'='*55}")
    log.info(f"  {split_name.upper()} RESULTS")
    log.info(f"{'='*55}")
    log.info(f"  Macro F1        : {macro_f1:.4f}")
    log.info(f"  Macro Precision : {macro_p :.4f}")
    log.info(f"  Macro Recall    : {macro_r :.4f}")
    log.info(f"{'='*55}\n")

    pd.DataFrame(per_query).to_csv(OUTPUT / f"{split_name}_per_query_metrics.csv", index=False)
    with open(OUTPUT / f"{split_name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Metrics saved -> {OUTPUT}")
    return summary, per_query


# ══════════════════════════════════════════════════════════════════
# 11.  SUBMISSION
# ══════════════════════════════════════════════════════════════════

def generate_submission(pipeline, test_df):
    query_col = next(c for c in ["query", "question", "text"] if c in test_df.columns)
    id_col    = next((c for c in ["query_id", "id"] if c in test_df.columns), None)
    queries   = test_df[query_col].tolist()
    preds     = pipeline.batch_retrieve(queries)
    rows      = [{"query_id": test_df.iloc[i][id_col] if id_col else f"test_{i:03d}",
                  "predicted_citations": ";".join(pred)}
                 for i, pred in enumerate(preds)]
    pd.DataFrame(rows).to_csv(OUTPUT / "submission.csv", index=False)
    log.info(f"Submission saved -> {OUTPUT / 'submission.csv'}")


# ══════════════════════════════════════════════════════════════════
# 12.  DASHBOARD
# ══════════════════════════════════════════════════════════════════

def save_metrics_dashboard(train_history, val_summary, test_summary=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        GOLD = "#FFD700"; TEAL = "#00E5CC"; RED = "#FF4C4C"; GRAY = "#555"
        fig  = plt.figure(figsize=(18, 10), facecolor="#0f1117")
        gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        tw   = dict(color="white")
        epochs = range(1, len(train_history["train_loss"]) + 1)

        # Loss curve
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.set_facecolor("#1a1d27")
        ax1.plot(epochs, train_history["train_loss"], "-o", color=TEAL, lw=2, label="Train Loss")
        ax1.plot(epochs, train_history["val_loss"],   "-s", color=GOLD, lw=2, label="Val Loss")
        ax1.set_title("Contrastive Training Loss", **tw, fontsize=13)
        ax1.set_xlabel("Epoch", **tw); ax1.set_ylabel("Loss", **tw)
        ax1.tick_params(colors="white")
        ax1.legend(facecolor="#1a1d27", labelcolor="white")
        for sp in ax1.spines.values(): sp.set_edgecolor(GRAY)

        # Val bar chart
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor("#1a1d27")
        mets = ["Macro F1", "Precision", "Recall"]
        vals = [val_summary["macro_f1"], val_summary["macro_precision"], val_summary["macro_recall"]]
        bars = ax2.bar(mets, vals, color=[TEAL, GOLD, RED], width=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                     f"{v:.3f}", ha="center", color="white", fontsize=10)
        ax2.set_ylim(0, 1.15); ax2.set_title("Validation Metrics", **tw, fontsize=13)
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values(): sp.set_edgecolor(GRAY)

        # Loss table
        ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor("#1a1d27"); ax3.axis("off")
        tbl_data = [[e, f"{tl:.4f}", f"{vl:.4f}"]
                    for e, tl, vl in zip(epochs, train_history["train_loss"], train_history["val_loss"])]
        tbl = ax3.table(cellText=tbl_data, colLabels=["Epoch", "Train Loss", "Val Loss"],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#1a1d27" if r > 0 else "#2a2d3d")
            cell.set_text_props(color="white"); cell.set_edgecolor(GRAY)
        ax3.set_title("Loss Per Epoch", **tw, fontsize=11)

        # Summary card
        ax4 = fig.add_subplot(gs[1, 1:]); ax4.set_facecolor("#1a1d27"); ax4.axis("off")
        lines = [
            "── FINAL RESULTS ──────────────────────────────",
            f"  Val  Macro F1        : {val_summary['macro_f1']:.4f}",
            f"  Val  Macro Precision : {val_summary['macro_precision']:.4f}",
            f"  Val  Macro Recall    : {val_summary['macro_recall']:.4f}",
        ]
        if test_summary:
            lines += ["",
                f"  Test Macro F1        : {test_summary.get('macro_f1','N/A')}",
                f"  Test Macro Precision : {test_summary.get('macro_precision','N/A')}",
                f"  Test Macro Recall    : {test_summary.get('macro_recall','N/A')}",
            ]
        lines += ["",
            f"  Model  : {CFG['embed_model'].split('/')[-1]}",
            f"  Device : {DEVICE}  |  FP16: {CFG['fp16']}",
            f"  Epochs : {CFG['epochs']}  |  LR: {CFG['lr']}",
            f"  Train batch : {CFG['train_batch']} x {CFG['grad_accum_steps']} accum = "
            f"{CFG['train_batch']*CFG['grad_accum_steps']} effective",
            f"  SeqLen train/infer: {CFG['max_seq_len']} / {CFG['infer_seq_len']}",
        ]
        ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes,
                 fontsize=9, color="white", va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2d3d", edgecolor=TEAL))
        ax4.set_title("Project Summary", **tw, fontsize=11)

        fig.suptitle("Swiss Legal Information Retrieval — Final Year Project",
                     color=GOLD, fontsize=15, y=0.98)
        out_path = OUTPUT / "metrics_dashboard.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"Dashboard saved -> {out_path}")
    except Exception as e:
        log.warning(f"Dashboard skipped (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════
# 13.  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Step 1: Load data ─────────────────────────────────────────
    dfs = load_data()

    # ── Step 2: Build corpus ──────────────────────────────────────
    citations, corpus_texts = build_corpus(dfs)
    if not citations:
        raise ValueError("Corpus is empty — check CSV paths and column names.")

    # ── Step 3: BM25 ─────────────────────────────────────────────
    bm25 = build_bm25(corpus_texts)

    # ── Step 4: Embedding model ───────────────────────────────────
    embed_model = EmbeddingModel(CFG["embed_model"])

    # ── Step 5: Fine-tune ─────────────────────────────────────────
    train_history = {"train_loss": [], "val_loss": []}
    if "train" in dfs and "val" in dfs:
        train_history = fine_tune_embedder(
            embed_model, dfs["train"], dfs["val"], citations, corpus_texts
        )
        best_dir = str(OUTPUT / "best_embed_model")
        log.info(f"Reloading best checkpoint from {best_dir} ...")
        embed_model.model = AutoModel.from_pretrained(best_dir).to(DEVICE)
        embed_model.model.eval()
    else:
        log.info("Skipping fine-tuning (train/val not found).")

    # ── Step 6: Encode corpus ─────────────────────────────────────
    emb_cache = OUTPUT / "corpus_embeddings.npy"
    if emb_cache.exists():
        log.info(f"Loading cached corpus embeddings from {emb_cache} ...")
        corpus_embs = np.load(str(emb_cache))
        log.info(f"Loaded shape: {corpus_embs.shape}")
    else:
        log.info("Encoding entire corpus (batch=16, will take ~30-60 min on GTX 1650) ...")
        corpus_embs = embed_model.encode(corpus_texts,
                                         batch_size=CFG["infer_batch"],
                                         seq_len=CFG["infer_seq_len"])
        np.save(str(emb_cache), corpus_embs)
        log.info(f"Corpus embeddings saved — shape: {corpus_embs.shape}")

    # ── Step 7: Reranker ──────────────────────────────────────────
    reranker = CrossEncoderReranker(CFG["rerank_model"])

    # ── Step 8: Build pipeline ────────────────────────────────────
    pipeline = LegalRetrievalPipeline(
        citations, corpus_texts, bm25, embed_model, corpus_embs, reranker
    )

    # ── Step 9: Evaluate ──────────────────────────────────────────
    val_summary  = None
    test_summary = None

    if "val" in dfs:
        val_summary, _ = evaluate_pipeline(pipeline, dfs["val"], split_name="val")

    if "test" in dfs:
        gold_col = next((c for c in ["gold_citations", "citations", "answer"]
                         if c in dfs["test"].columns), None)
        if gold_col:
            test_summary, _ = evaluate_pipeline(pipeline, dfs["test"], split_name="test")
        else:
            generate_submission(pipeline, dfs["test"])

    # ── Step 10: Dashboard + config ───────────────────────────────
    if val_summary:
        save_metrics_dashboard(train_history, val_summary, test_summary)

    with open(OUTPUT / "config.json", "w") as fh:
        json.dump({**CFG, "device": str(DEVICE)}, fh, indent=2)

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime : {elapsed/60:.1f} min")
    log.info(f"All outputs   : {OUTPUT}")


# ── Windows multiprocessing guard (REQUIRED) ─────────────────────
if __name__ == "__main__":
    main()