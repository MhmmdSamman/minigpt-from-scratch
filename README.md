# 🧠 MiniGPT — GPT from Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**Implementasi GPT-style Transformer dari nol menggunakan PyTorch murni. Dilatih pada dataset WikiText-2 dengan GPT-2 tokenizer.**

[Arsitektur](#arsitektur) · [Instalasi](#instalasi) · [Training](#training) · [Inferensi](#inferensi)

</div>

---

## ✨ Fitur Utama

- 🏗 **Transformer dari nol** — tidak menggunakan library high-level
- 🔥 **Multi-Head Causal Self-Attention** dengan causal mask
- ⚡ **Gradient Accumulation** — efisien untuk GPU VRAM terbatas
- 📉 **Cosine Annealing LR Scheduler**
- 💾 **Checkpoint system** — save best model & resume training
- 📊 **Perplexity tracking** di setiap epoch
- 🌐 **Inferensi Bahasa Indonesia** via auto-translate

---

## 🗂 Struktur Proyek

```
minigpt-from-scratch/
│
├── model.py            # Arsitektur MiniGPT (MHA, FFN, TransformerBlock)
├── train.py            # Training loop + evaluasi + checkpoint
├── evaluate.py         # Evaluasi model: val loss & perplexity
├── generate.py         # Text generation + inferensi Bahasa Indonesia
│
├── data/
│   └── tokenized/      # Dataset WikiText-2 yang sudah ditokenisasi
│       └── .gitkeep
│
├── checkpoints/
│   ├── best_model.pt   # Model terbaik (val loss terendah)
│   └── last_model.pt   # Model epoch terakhir
│
├── notebooks/
│   ├── llm.ipynb       # Notebook eksperimen bertahap
│   └── minigpt.ipynb   # Notebook lengkap dengan penjelasan
│
├── assets/
│   └── architecture.png
│
├── requirements.txt
└── .gitignore
```

---

## 🏗 Arsitektur

```
Input Tokens (seq_len)
        ↓
Token Embedding [vocab=50257, dim=128]
        +
Positional Embedding [max_len=512, dim=128]
        ↓
┌─────────────────────────────┐
│   Transformer Block × 4     │
│  ┌───────────────────────┐  │
│  │  Layer Norm           │  │
│  │  Multi-Head Attention │  │  ← 4 heads, causal mask
│  │  + Dropout(0.1)       │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Layer Norm           │  │
│  │  Feed-Forward (GELU)  │  │  ← dim × 4 → dim
│  │  + Dropout(0.1)       │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
        ↓
   Layer Norm
        ↓
  LM Head [dim → vocab]
        ↓
   Logits / Loss
```

### Konfigurasi Model

| Parameter          | Nilai   |
|-------------------|---------|
| Vocab Size        | 50,257  |
| Embedding Dim     | 128     |
| Num Heads         | 4       |
| Num Layers        | 4       |
| Max Seq Length    | 512     |
| Total Parameters  | ~7.5M   |
| Model Size        | ~28 MB  |

---

## ⚙️ Instalasi

```bash
# Clone repo
git clone https://github.com/MuhammadSamman/minigpt-from-scratch.git
cd minigpt-from-scratch

# Buat virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
# Step 1: Download & tokenisasi dataset WikiText-2
python -c "
from datasets import load_dataset
from transformers import AutoTokenizer
import os

os.makedirs('data', exist_ok=True)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])
tokenized.save_to_disk('data/tokenized')
print('Dataset siap!')
"

# Step 2: Train model
python train.py

# Resume training dari checkpoint
python train.py --resume checkpoints/best_model.pt
```

### Training Config

| Parameter           | Nilai  |
|--------------------|--------|
| Batch Size         | 16     |
| Epochs             | 5      |
| Learning Rate      | 3e-4   |
| Grad Accumulation  | 4      |
| Optimizer          | AdamW  |
| Scheduler          | CosineAnnealing |

---

## 💬 Inferensi

```bash
# Generate teks Bahasa Inggris
python generate.py --prompt "The history of artificial intelligence"

# Generate dari Bahasa Indonesia (auto-translate)
python generate.py --prompt "kecerdasan buatan adalah" --lang id --max_tokens 50
```

---

## 📊 Evaluasi

```bash
python evaluate.py
# Output:
# Val Loss   : 4.2310
# Perplexity : 68.73
```

---

## 📦 Requirements

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
deep-translator>=1.11.0
numpy>=1.24.0
```

---

## 📄 Lisensi

MIT License — lihat [LICENSE](LICENSE)

---

<div align="center">
Made with ❤️ by <a href="https://github.com/MuhammadSamman">Muhammad Samman</a>
</div>
