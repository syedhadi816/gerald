# Gerald

**A Retrieval and Generative Ensemble for Conversational AI**

[![Try on Kaggle](https://img.shields.io/badge/Try%20Gerald-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/syedhadi816/gerald-a-conversational-ai)

Gerald is an open-source, open-domain conversational AI that combines a **retrieval-based** system (BERT) with a **generation-based** system (T5) in a sequential ensemble—delivering contextually relevant, structurally coherent responses that outperform either approach alone.

![Gerald Salton – The Father of Information Retrieval](https://github.com/user-attachments/assets/cbab2f65-aa97-413e-b5b5-7d0faee9ec54)

Named after [Gerald Salton](https://en.wikipedia.org/wiki/Gerard_Salton), a pioneer in information retrieval.

---

## What is Gerald?

Gerald uses a **retrieve-then-generate** pipeline:

1. **Retrieval** — A BERT-based sentence encoder scores your query against a large Q&A and conversation dataset. The top-**k** most similar question–answer pairs are retrieved.
2. **Generation** — A T5 conditional generator takes the retrieved context plus your query and produces a new, fluent response that stays on-topic and in style.

This hybrid design gives you the reliability of retrieval (grounded in real data) and the flexibility of generation (natural, varied answers).

---

## Architecture

| Component | Role |
|-----------|------|
| **BERT** (sentence-transformers) | Encodes user query and dataset questions; used for semantic similarity (e.g. cosine) to get top-**k** candidates. |
| **PCA** | Optional dimensionality reduction for efficient similarity search over question embeddings. |
| **T5** (Hugging Face) | Conditional text generation: input = retrieved context + user query; output = generated reply. |

The retrieval model is trained on large-scale conversational Q&A and dialogue data; the generative model conditions on that retrieved text to produce contextually and structurally aligned responses.

---

## Try Gerald

You can run and chat with Gerald in a ready-to-use environment here:

**[→ Try Gerald on Kaggle](https://www.kaggle.com/code/syedhadi816/gerald-a-conversational-ai)**

No local setup required—run the notebook and start a conversation.

---

## Project structure

```
gerald/
├── README.md
├── requirements.txt
└── gerald.py          # Core setup: BERT encoder, T5 model, data loading (Kaggle paths)
```

The full interactive pipeline (similarity search, conditioning, and generation) is implemented in the [Kaggle notebook](https://www.kaggle.com/code/syedhadi816/gerald-a-conversational-ai); `gerald.py` contains the shared model and data-loading logic used there.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies (e.g. `transformers`, `sentence-transformers`, `tensorflow`/`keras`, `scikit-learn`, `nltk`).

For running locally, you’ll need the same data inputs as in the Kaggle notebook (e.g. `q_a_pairs.csv`, `q_pca_embed.txt`, `pca.pkl`) and to adjust paths in `gerald.py` from `/kaggle/input/...` to your local paths.

---

## Citation & acknowledgments

- **BERT** / sentence representations: [Sentence-BERT](https://www.sbert.net/)
- **T5**: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (Raffel et al.)
- Gerald is named after Gerald Salton for his foundational work in information retrieval.

---

## License

This project is open source. See the repository for license details.

---

**Built with BERT and T5 · Try it on [Kaggle](https://www.kaggle.com/code/syedhadi816/gerald-a-conversational-ai)**
