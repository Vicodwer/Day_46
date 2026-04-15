# Week 08 - Wednesday: CNNs + Embeddings

**PG Diploma - AI-ML & Agentic AI Engineering - IIT Gandhinagar**

---

## What this notebook covers

| Sub-step | Difficulty | Description |
|---|---|---|
| 1 | Easy | EDA of `social_media_posts.csv` - class distributions, imbalance analysis |
| 2 | Easy | MNIST characterisation and normalised DataLoader setup |
| 3 | Medium | CNN architecture, training, filter visualisation |
| 4 | Medium | Hate speech classifier + SBERT semantic search index |
| 5 | Medium | Two-stage moderation pipeline + evaluation and recommendation |
| 6 | Hard | Empirical TF-IDF vs SBERT comparison with analysis |
| 7 | Hard | MNIST CNN transfer experiment on social media data |

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd week-08/wednesday
```

### 2. Python version

Python **3.10** or later is required.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Download `social_media_posts.csv` from the LMS and place it in:

```
week-08/wednesday/social_media_posts.csv
```

MNIST is downloaded automatically via `torchvision` on first run.

### 5. Launch the notebook

```bash
jupyter notebook W8_Wednesday_CNNs_Embeddings.ipynb
```

Run all cells in order from top to bottom (**Kernel > Restart & Run All**).

---

## Dependencies

```
torch>=2.0
torchvision>=0.15
sentence-transformers>=2.2
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
Pillow>=9.0
jupyter>=1.0
```

Install everything at once:

```bash
pip install torch torchvision sentence-transformers scikit-learn pandas numpy matplotlib seaborn Pillow jupyter
```

---

## Outputs produced

| File | Description |
|---|---|
| `class_distributions.png` | Bar charts of hate_speech and spam class imbalance |
| `mnist_samples.png` | Sample MNIST images per digit class |
| `cnn_training_curve.png` | Training loss and test accuracy curves |
| `conv1_filters.png` | Learned first-layer CNN filter visualisations |

---

## Repository structure

```
week-08/
  wednesday/
    W8_Wednesday_CNNs_Embeddings.ipynb
    README.md
    prompts.md
    requirements.txt
    social_media_posts.csv          <- download from LMS, not committed
```

---

## Notes

- `social_media_posts.csv` is not committed (contains potentially sensitive content and is provided via LMS).
- No API keys or `.env` files are used.
- Random seed is fixed to `42` throughout for reproducibility.
- Sub-steps 6 and 7 (Hard) are included and attempted for Band 4 eligibility.
