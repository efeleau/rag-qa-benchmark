# Does retrieval help an LLM answer questions?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/efeleau/rag-qa-benchmark/blob/main/RAG.ipynb)

This project measures how much **retrieval-augmented generation (RAG)** improves an LLM's answers to factual questions, and compares four ways of retrieving passages: **BM25**, **dense retrieval**, **hybrid** (rank fusion of both) and **hybrid + cross-encoder re-ranking**. Differences between methods are tested statistically with a paired bootstrap, to separate real effects from sampling noise.

The notebook is available in two versions: [`RAG.ipynb`](RAG.ipynb) (French) and [`RAG_en.ipynb`](RAG_en.ipynb) (English), with the same code and results. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/efeleau/rag-qa-benchmark/blob/main/RAG_en.ipynb) (English version)

## Results

The same LLM (Qwen2.5-1.5B-Instruct) answered the same **300 questions**, without any passage (*closed-book*) and then with the top 5 passages of each retrieval method.

| Method | Exact Match | 95% CI (bootstrap) | F1 |
|---|---|---|---|
| Closed-book (no retrieval) | 0.320 | [0.263, 0.370] | 0.353 |
| BM25 | 0.553 | [0.497, 0.607] | 0.643 |
| Dense | 0.540 | [0.480, 0.597] | 0.617 |
| Hybrid (RRF) | 0.577 | [0.520, 0.630] | 0.656 |
| Hybrid + re-rank | **0.583** | [0.527, 0.637] | **0.670** |

Paired bootstrap on per-question Exact Match differences:

| Comparison (A − B) | Δ EM | 95% CI | Significant | A only correct | B only correct |
|---|---|---|---|---|---|
| BM25 − Closed-book | +0.233 | [+0.180, +0.290] | yes | 80 | 10 |
| Hybrid − BM25 | +0.023 | [−0.003, +0.050] | no | 12 | 5 |
| Hybrid − Dense | +0.037 | [+0.013, +0.063] | yes | 13 | 2 |
| BM25 − Dense | +0.013 | [−0.020, +0.047] | no | 15 | 11 |
| Hybrid + re-rank − Hybrid | +0.007 | [−0.020, +0.033] | no | 9 | 7 |

**Key findings**
1. **Retrieval sharply improves answers**: Exact Match rises from 0.32 to 0.58 (+26 points), far beyond sampling noise.
2. **Hybrid retrieval is the most reliable method**: it significantly beats dense retrieval (13 of 15 disagreements won) and leads BM25, although that lead is not yet significant.
3. **Re-ranking brings no measurable gain** on this corpus, despite an extra model and extra compute.
4. **Remaining errors are about content, not format**: the gap between "answer contained in the prediction" and Exact Match is only 2–3 points.

**Recommendation:** for this kind of corpus, hybrid BM25 + dense retrieval without re-ranking gives the best trade-off between answer quality and cost.

## Method

- **Data**: [`rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia), with 3,200 Wikipedia passages and 918 question/answer pairs. 300 questions were drawn at random with a fixed seed.
- **BM25**: implemented from scratch as an inverted index (IDF weighting, term-frequency saturation, length normalisation).
- **Dense retrieval**: `multi-qa-MiniLM-L6-cos-v1` embeddings, cosine similarity.
- **Hybrid**: Reciprocal Rank Fusion of the BM25 and dense rankings (top 100 of each).
- **Re-ranking**: `ms-marco-MiniLM-L-6-v2` cross-encoder applied to the top 20 hybrid candidates.
- **Generation**: `Qwen2.5-1.5B-Instruct`, greedy decoding, prompted for a short answer.
- **Metrics**: SQuAD-style Exact Match and token F1, plus a lenient "contains the answer" score.
- **Statistics**: bootstrap confidence intervals (2,000 resamples) and paired bootstrap for method comparisons.

## Limitations

- A single, small LLM (1.5B parameters): a larger model would score higher closed-book, and would probably gain less from retrieval.
- A small corpus (3,200 passages on related topics). Retrieval over millions of passages would be harder.
- A single value of k (5 passages), with no hyperparameter tuning.
- 300 questions: differences of 2–4 points between retrieval methods are close to the detection threshold, and five tests were run without correcting for multiple comparisons.
- Exact Match is strict: correct answers phrased differently are counted as wrong.

## How to run

Open the notebook in Colab with the badge above, select a **T4 GPU** (`Runtime → Change runtime type`) and run all cells. It takes about 15 minutes.

To run locally (an NVIDIA GPU is recommended):
```bash
pip install -r requirements.txt
jupyter notebook RAG.ipynb      # or RAG_en.ipynb for the English version
```

## Background

This project started as a course project of the Master's in Statistics & Machine Learning at Sorbonne Université (2024). It was rebuilt in 2026 on a public dataset, with an open model and a statistical evaluation, so that every result is reproducible.
