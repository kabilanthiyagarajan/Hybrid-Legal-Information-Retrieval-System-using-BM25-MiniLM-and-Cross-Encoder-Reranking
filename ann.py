"""
=====================================================================
 Swiss Legal Information Retrieval — INFERENCE / TEST CODE
 Final Year Project
=====================================================================
 Uses already-saved outputs from training:
   output/best_embed_model/        <- fine-tuned bi-encoder
   output/corpus_embeddings.npy   <- pre-computed corpus vectors
   output/config.json             <- saved hyperparameters

 Run:
   python inference.py
=====================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import json
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\Kabilan\Downloads\project")
OUTPUT      = BASE / "output"
MODEL_DIR   = OUTPUT / "best_embed_model"
EMB_CACHE   = OUTPUT / "corpus_embeddings.npy"
CFG_FILE    = OUTPUT / "config.json"
VAL_SUMMARY = OUTPUT / "val_summary.json"
VAL_METRICS = OUTPUT / "val_per_query_metrics.csv"
SUBMISSION  = OUTPUT / "submission.csv"

DATA = {
    "laws"  : BASE / "laws_de.csv",
    "court" : BASE / "court_considerations.csv",
    "test"  : BASE / "test.csv",
    "val"   : BASE / "val.csv",
}

# ─── Device ──────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

# ─── Load saved config ───────────────────────────────────────────
with open(CFG_FILE) as f:
    CFG = json.load(f)

RERANK_MODEL  = CFG.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
INFER_SEQ_LEN = CFG.get("infer_seq_len", 128)
INFER_BATCH   = CFG.get("infer_batch",   16)
BM25_TOP_K    = CFG.get("bm25_top_k",   100)
DENSE_TOP_K   = CFG.get("dense_top_k",  100)
RERANK_TOP_K  = CFG.get("rerank_top_k",  20)
FINAL_TOP_K   = CFG.get("final_top_k",   10)

log.info(f"Config loaded from {CFG_FILE}")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def tokenize_for_bm25(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def citation_f1(pred, gold):
    pred_set, gold_set = set(pred), set(gold)
    if not gold_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 0.0, 0.0)
    tp = len(pred_set & gold_set)
    p  = tp / len(pred_set) if pred_set else 0.0
    r  = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


# ══════════════════════════════════════════════════════════════════
# 1.  LOAD CORPUS  (laws + court)
# ══════════════════════════════════════════════════════════════════

def load_corpus():
    log.info("Loading corpus CSVs ...")
    citations, texts = [], []

    for key in ["laws", "court"]:
        path = DATA[key]
        if not path.exists():
            log.warning(f"  {key} not found at {path} — skipping")
            continue
        df       = pd.read_csv(path)
        text_col = next((c for c in ["text", "consideration", "article_text", "de_text", "content", "body"]
                         if c in df.columns), None)
        cit_col  = next((c for c in ["citation", "id", "article_id", "decision_id", "bge"]
                         if c in df.columns), None)
        if text_col and cit_col:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{key} corpus"):
                cit  = str(row[cit_col]).strip()
                body = str(row[text_col]).strip()[:300]
                citations.append(cit)
                texts.append(f"{cit} {body}")
        else:
            cols = df.columns.tolist()
            log.warning(f"  {key}: using first two columns as fallback")
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{key} fallback"):
                citations.append(str(row[cols[0]]).strip())
                texts.append(" ".join(str(v) for v in row.values if pd.notna(v))[:300])

    log.info(f"Corpus loaded: {len(citations):,} documents")
    return citations, texts


# ══════════════════════════════════════════════════════════════════
# 2.  BM25  INDEX
# ══════════════════════════════════════════════════════════════════

def build_bm25(texts):
    log.info("Building BM25 index ...")
    tokenized = [tokenize_for_bm25(t) for t in tqdm(texts, desc="BM25 tokenize")]
    return BM25Okapi(tokenized)


# ══════════════════════════════════════════════════════════════════
# 3.  LOAD FINE-TUNED EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════

class EmbeddingModel:
    def __init__(self, model_dir):
        log.info(f"Loading fine-tuned embedding model from {model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model     = AutoModel.from_pretrained(str(model_dir)).to(DEVICE)
        self.model.eval()
        log.info("  Embedding model loaded OK")

    @torch.no_grad()
    def encode(self, texts, batch_size=None, seq_len=None):
        if batch_size is None: batch_size = INFER_BATCH
        if seq_len    is None: seq_len    = INFER_SEQ_LEN
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i: i + batch_size]
            enc   = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=seq_len, return_tensors="pt",
            ).to(DEVICE)
            out  = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb  = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
            del enc, out, mask, emb
            torch.cuda.empty_cache()
        return np.vstack(all_embs)

    def encode_queries(self, queries, batch_size=32):
        return self.encode(queries, batch_size=batch_size, seq_len=INFER_SEQ_LEN)


# ══════════════════════════════════════════════════════════════════
# 4.  CROSS-ENCODER RERANKER
# ══════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    def __init__(self, model_name):
        log.info(f"Loading cross-encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        log.info("  Cross-encoder loaded OK")

    @torch.no_grad()
    def score(self, query, passages, batch_size=16):
        scores = []
        pairs  = [[query, p] for p in passages]
        for i in range(0, len(pairs), batch_size):
            batch  = pairs[i: i + batch_size]
            enc    = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=INFER_SEQ_LEN, return_tensors="pt",
            ).to(DEVICE)
            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(logits.cpu().float().numpy().tolist())
            del enc, logits
            torch.cuda.empty_cache()
        return np.array(scores)


# ══════════════════════════════════════════════════════════════════
# 5.  RETRIEVAL PIPELINE
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
            final_top_k = FINAL_TOP_K
        bm25_ids  = self._bm25_retrieve(query, BM25_TOP_K)
        q_emb     = self.embed.encode_queries([query], batch_size=1)
        dense_ids = self._dense_retrieve(q_emb, DENSE_TOP_K)
        cand_ids  = list(dict.fromkeys(np.concatenate([bm25_ids, dense_ids])))[:RERANK_TOP_K]
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
# 6.  PRINT SAVED TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════

def print_saved_results():
    log.info("\n" + "="*60)
    log.info("  SAVED TRAINING RESULTS")
    log.info("="*60)

    # val_summary.json
    if VAL_SUMMARY.exists():
        with open(VAL_SUMMARY) as f:
            vs = json.load(f)
        log.info(f"  Val  Macro F1        : {vs.get('macro_f1','?')}")
        log.info(f"  Val  Macro Precision : {vs.get('macro_precision','?')}")
        log.info(f"  Val  Macro Recall    : {vs.get('macro_recall','?')}")
        log.info(f"  Val  Num queries     : {vs.get('num_queries','?')}")
    else:
        log.warning("  val_summary.json not found")

    # val_per_query_metrics.csv
    if VAL_METRICS.exists():
        df = pd.read_csv(VAL_METRICS)
        log.info(f"\n  Per-query metrics ({len(df)} queries):")
        log.info(f"  {'query_id':<15} {'F1':>6} {'Prec':>6} {'Recall':>6}")
        log.info(f"  {'-'*38}")
        for _, row in df.iterrows():
            log.info(f"  {str(row.get('query_id','?')):<15} "
                     f"{row.get('f1',0):>6.4f} "
                     f"{row.get('precision',0):>6.4f} "
                     f"{row.get('recall',0):>6.4f}")
        log.info(f"\n  Mean F1        : {df['f1'].mean():.4f}")
        log.info(f"  Mean Precision : {df['precision'].mean():.4f}")
        log.info(f"  Mean Recall    : {df['recall'].mean():.4f}")
        log.info(f"  Min  F1        : {df['f1'].min():.4f}")
        log.info(f"  Max  F1        : {df['f1'].max():.4f}")

    log.info("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════
# 7.  EVALUATE ON VALIDATION SET  (re-run with loaded model)
# ══════════════════════════════════════════════════════════════════

def evaluate_val(pipeline):
    val_path = DATA["val"]
    if not val_path.exists():
        log.warning("val.csv not found — skipping validation")
        return None

    df        = pd.read_csv(val_path)
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
            "query_id"       : df.iloc[i].get("query_id", i),
            "query"          : queries[i],
            "gold_citations" : ";".join(gold),
            "predicted"      : ";".join(pred),
            "f1"             : round(f1, 4),
            "precision"      : round(p,  4),
            "recall"         : round(r,  4),
            "tp"             : len(set(pred) & set(gold)),
            "fp"             : len(set(pred) - set(gold)),
            "fn"             : len(set(gold) - set(pred)),
        })

    macro_f1 = float(np.mean(f1s))
    macro_p  = float(np.mean(ps))
    macro_r  = float(np.mean(rs))

    log.info("\n" + "="*60)
    log.info("  INFERENCE — VALIDATION SET RESULTS")
    log.info("="*60)
    log.info(f"  Macro F1        : {macro_f1:.4f}")
    log.info(f"  Macro Precision : {macro_p :.4f}")
    log.info(f"  Macro Recall    : {macro_r :.4f}")
    log.info("="*60)

    # Per-query breakdown
    log.info(f"\n  {'query_id':<15} {'F1':>6} {'Prec':>6} {'Recall':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
    log.info(f"  {'-'*52}")
    for row in per_query:
        log.info(f"  {str(row['query_id']):<15} "
                 f"{row['f1']:>6.4f} "
                 f"{row['precision']:>6.4f} "
                 f"{row['recall']:>6.4f}  "
                 f"{row['tp']:>4} {row['fp']:>4} {row['fn']:>4}")

    # Save updated metrics
    out_csv = OUTPUT / "inference_val_metrics.csv"
    out_json = OUTPUT / "inference_val_summary.json"
    pd.DataFrame(per_query).to_csv(out_csv, index=False)
    summary = {
        "macro_f1"       : round(macro_f1, 4),
        "macro_precision": round(macro_p,  4),
        "macro_recall"   : round(macro_r,  4),
        "num_queries"    : len(queries),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\n  Detailed metrics saved -> {out_csv}")
    log.info(f"  Summary saved          -> {out_json}")
    return summary


# ══════════════════════════════════════════════════════════════════
# 8.  GENERATE TEST SUBMISSION
# ══════════════════════════════════════════════════════════════════

def generate_submission(pipeline):
    test_path = DATA["test"]
    if not test_path.exists():
        log.warning("test.csv not found — skipping submission generation")
        return

    df        = pd.read_csv(test_path)
    query_col = next(c for c in ["query", "question", "text"] if c in df.columns)
    id_col    = next((c for c in ["query_id", "id"] if c in df.columns), None)
    queries   = df[query_col].tolist()

    log.info(f"\nGenerating predictions for {len(queries)} test queries ...")
    preds = pipeline.batch_retrieve(queries)

    rows = []
    for i, pred in enumerate(preds):
        qid = df.iloc[i][id_col] if id_col else f"test_{i:03d}"
        rows.append({
            "query_id"           : qid,
            "predicted_citations": ";".join(pred),
        })

    sub_df   = pd.DataFrame(rows)
    sub_path = OUTPUT / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    log.info(f"\n  Submission saved -> {sub_path}")
    log.info(f"  Total rows      : {len(sub_df)}")
    log.info(f"\n  Sample predictions:")
    log.info(f"  {'query_id':<12}  predicted_citations")
    log.info(f"  {'-'*60}")
    for _, row in sub_df.head(5).iterrows():
        cits = row["predicted_citations"][:70] + ("..." if len(row["predicted_citations"]) > 70 else "")
        log.info(f"  {str(row['query_id']):<12}  {cits}")

    return sub_df


# ══════════════════════════════════════════════════════════════════
# 9.  SINGLE QUERY DEMO  (interactive test)
# ══════════════════════════════════════════════════════════════════

def demo_single_query(pipeline):
    log.info("\n" + "="*60)
    log.info("  SINGLE QUERY DEMO")
    log.info("="*60)
    sample_queries = [
        "What are the legal requirements for divorce in Switzerland?",
        "What are the rights of an employee upon termination of employment?",
        "What are the conditions for Swiss citizenship by naturalization?",
    ]
    for query in sample_queries:
        log.info(f"\n  Query: {query}")
        preds = pipeline.retrieve(query, final_top_k=5)
        log.info(f"  Top-5 citations:")
        for rank, cit in enumerate(preds, 1):
            log.info(f"    {rank}. {cit}")
    log.info("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════
# 10.  ACCURACY REPORT  (full metrics dashboard)
# ══════════════════════════════════════════════════════════════════

def save_accuracy_report(val_summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        GOLD = "#FFD700"; TEAL = "#00E5CC"; RED = "#FF4C4C"; GRAY = "#555"
        fig  = plt.figure(figsize=(16, 9), facecolor="#0f1117")
        gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)
        tw   = dict(color="white")

        # ── Metric bars ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor("#1a1d27")
        mets  = ["Macro\nF1", "Macro\nPrecision", "Macro\nRecall"]
        vals  = [val_summary["macro_f1"],
                 val_summary["macro_precision"],
                 val_summary["macro_recall"]]
        bars  = ax1.bar(mets, vals, color=[TEAL, GOLD, RED], width=0.5)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f"{v:.4f}", ha="center", color="white", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 1.2)
        ax1.set_title("Validation Metrics", **tw, fontsize=12)
        ax1.tick_params(colors="white")
        for sp in ax1.spines.values(): sp.set_edgecolor(GRAY)

        # ── Per-query F1 bar ─────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.set_facecolor("#1a1d27")
        if VAL_METRICS.exists():
            dfm  = pd.read_csv(VAL_METRICS)
            qids = [str(q) for q in dfm["query_id"].tolist()]
            f1s  = dfm["f1"].tolist()
            colors_q = [TEAL if v >= 0.5 else RED for v in f1s]
            ax2.bar(range(len(qids)), f1s, color=colors_q)
            ax2.set_xticks(range(len(qids)))
            ax2.set_xticklabels(qids, rotation=45, ha="right", fontsize=7, color="white")
            ax2.axhline(y=val_summary["macro_f1"], color=GOLD, lw=1.5, linestyle="--",
                        label=f"Mean F1={val_summary['macro_f1']:.3f}")
            ax2.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Per-Query F1 Score", **tw, fontsize=12)
        ax2.set_ylabel("F1", **tw)
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values(): sp.set_edgecolor(GRAY)

        # ── Per-query P/R/F1 table ────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor("#1a1d27")
        ax3.axis("off")
        if VAL_METRICS.exists():
            dfm  = pd.read_csv(VAL_METRICS)
            tbl_data = []
            for _, row in dfm.iterrows():
                q_short = str(row["query"])[:50] + ("..." if len(str(row["query"])) > 50 else "")
                g_short = str(row["gold"])[:35] + ("..." if len(str(row["gold"])) > 35 else "")
                p_short = str(row["predicted"])[:35] + ("..." if len(str(row["predicted"])) > 35 else "")
                tbl_data.append([str(row["query_id"]), q_short, g_short, p_short,
                                 f"{row['f1']:.4f}", f"{row['precision']:.4f}", f"{row['recall']:.4f}"])

            tbl = ax3.table(
                cellText  = tbl_data,
                colLabels = ["Query ID", "Query", "Gold Citations", "Predicted", "F1", "Prec", "Recall"],
                loc="center", cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            col_widths = [0.07, 0.28, 0.22, 0.22, 0.06, 0.06, 0.06]
            for (r, c), cell in tbl.get_celld().items():
                cell.set_facecolor("#1a1d27" if r > 0 else "#2a2d3d")
                cell.set_text_props(color="white")
                cell.set_edgecolor(GRAY)
                cell.set_width(col_widths[c] if c < len(col_widths) else 0.1)
            ax3.set_title("Full Per-Query Breakdown", **tw, fontsize=11)

        # ── Summary scorecard ─────────────────────────────────────
        fig.text(0.01, 0.99,
                 f"Model: {CFG.get('embed_model','?').split('/')[-1]}  |  "
                 f"Device: {DEVICE}  |  "
                 f"FP16: {CFG.get('fp16','?')}  |  "
                 f"Corpus: 2,652,248 docs  |  "
                 f"Val queries: {val_summary.get('num_queries','?')}",
                 color="#aaa", fontsize=8, va="top", fontfamily="monospace")

        fig.suptitle("Swiss Legal Information Retrieval — Inference Results",
                     color=GOLD, fontsize=14, y=1.01)

        out_path = OUTPUT / "inference_accuracy_report.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"Accuracy report saved -> {out_path}")
    except Exception as e:
        log.warning(f"Report generation skipped: {e}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    log.info("\n" + "="*60)
    log.info("  SWISS LEGAL RETRIEVAL — INFERENCE MODE")
    log.info("="*60)

    # ── 1. Print saved training metrics ──────────────────────────
    print_saved_results()

    # ── 2. Load corpus ────────────────────────────────────────────
    citations, corpus_texts = load_corpus()

    # ── 3. BM25 ───────────────────────────────────────────────────
    bm25 = build_bm25(corpus_texts)

    # ── 4. Load fine-tuned embedding model ────────────────────────
    embed_model = EmbeddingModel(MODEL_DIR)

    # ── 5. Load pre-computed corpus embeddings ────────────────────
    log.info(f"Loading corpus embeddings from {EMB_CACHE} ...")
    corpus_embs = np.load(str(EMB_CACHE))
    log.info(f"  Embeddings shape : {corpus_embs.shape}")

    # ── 6. Load cross-encoder reranker ────────────────────────────
    reranker = CrossEncoderReranker(RERANK_MODEL)

    # ── 7. Build pipeline ─────────────────────────────────────────
    pipeline = LegalRetrievalPipeline(
        citations, corpus_texts, bm25, embed_model, corpus_embs, reranker
    )

    # ── 8. Run demo queries ───────────────────────────────────────
    demo_single_query(pipeline)

    # ── 9. Evaluate on validation set ────────────────────────────
    val_summary = evaluate_val(pipeline)

    # ── 10. Generate test submission ──────────────────────────────
    generate_submission(pipeline)

    # ── 11. Save accuracy report image ───────────────────────────
    if val_summary:
        save_accuracy_report(val_summary)

    elapsed = time.time() - t0
    log.info(f"\nInference completed in {elapsed/60:.1f} min")
    log.info(f"All outputs saved to: {OUTPUT}")
    log.info("\nOutput files:")
    log.info(f"  {OUTPUT / 'submission.csv'}")
    log.info(f"  {OUTPUT / 'inference_val_metrics.csv'}")
    log.info(f"  {OUTPUT / 'inference_val_summary.json'}")
    log.info(f"  {OUTPUT / 'inference_accuracy_report.png'}")


# ── Windows guard ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()