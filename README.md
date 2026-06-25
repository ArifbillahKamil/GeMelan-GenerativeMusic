# 🎵 GeMelan — Folk Melody Generator

> Aplikasi Generative AI yang menghasilkan melodi musik folk secara otomatis menggunakan model LSTM yang dibangun dari scratch dengan PyTorch.

---

## 👥 Anggota Kelompok

| Nama | NIM | Kontribusi |
|------|-----|------------|
| Muhammad Arifbillah Kamil | 1203230028 | Data & Preprocessing |
| M Iqbal Ilham Prabowo | 1203230088 | Arsitektur Model & Training |
| Jehova Putra Yan Nehru | 1203230107 | Inference, Evaluasi & Interface |

**Kelas:** IF-03-01
**Mata Kuliah:** Generative AI
**Link Repository:** [https://github.com/ArifbillahKamil/GeMelan](https://github.com/ArifbillahKamil/GeMelan)
**Link Demo:** [akan diisi setelah deploy]

---

## 📌 Deskripsi Proyek

**GeMelan** (Generative Melody) adalah sistem generatif berbasis **LSTM (Long Short-Term Memory)** yang dibangun dari scratch menggunakan PyTorch. Model belajar pola melodi dari **Nottingham Music Database** — koleksi 1.034 lagu folk dalam format MIDI — kemudian menghasilkan melodi baru yang orisinal dalam format `.mid` yang dapat diputar dan didownload.

### Fitur Utama
- 🎼 Generate melodi baru dari seed note atau secara acak
- 🎛️ Kontrol parameter: panjang melodi, temperature, top-k, dan tempo
- 💾 Download hasil generate dalam format MIDI (`.mid`)
- 📊 Visualisasi piano roll dari melodi yang dihasilkan
- 🖥️ Antarmuka web interaktif berbasis Gradio

> ✅ **Tidak menggunakan provider LLM/API tertutup** (OpenAI, Anthropic, Google Gemini, dsb.)
> Model dibangun sepenuhnya dari scratch menggunakan PyTorch tanpa bergantung pada layanan eksternal.

---

## 🗂️ Struktur Folder

```
GeMelan/
├── data/
│   ├── nottingham-source/      # Dataset Nottingham (hasil clone)
│   │   └── MIDI/
│   │       └── melody/         # 1.034 file MIDI melody (yang digunakan)
│   └── processed/              # Hasil preprocessing (otomatis dibuat)
│       ├── train.npz           # Data training (80%)
│       ├── val.npz             # Data validasi (10%)
│       ├── test.npz            # Data testing (10%)
│       ├── vocab.pkl           # Vocabulary token
│       └── vocab_meta.json     # Metadata vocabulary
├── src/
│   ├── model/
│   │   ├── lstm.py             # Arsitektur MelodyLSTM
│   │   └── train.py            # Training loop
│   ├── preprocessing/
│   │   └── midi_parser.py      # Parser, tokenizer, windowing, split
│   ├── evaluation/
│   │   └── metrics.py          # Perplexity, diversity, repetition, KL divergence
│   └── inference/
│       └── generate.py         # Temperature & top-k sampling, output MIDI
├── notebooks/                  # Eksplorasi & eksperimen Jupyter
├── models/                     # Checkpoint model hasil training
│   └── best_model.pt           # Model terbaik (berdasarkan val loss)
├── outputs/
│   ├── midi/                   # Hasil generate melodi (.mid)
│   ├── plots/                  # Piano roll, training curve, pitch distribution
│   └── logs/                   # Log training JSON, evaluation report
├── app/
│   └── app.py                  # Gradio interface
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Instalasi

### Prasyarat
- Python 3.9+
- pip

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/ArifbillahKamil/GeMelan.git
cd GeMelan

# 2. Buat virtual environment
python -m venv venv

# Aktivasi (Linux/Mac)
source venv/bin/activate

# Aktivasi (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Salin file environment
cp .env.example .env
```

---

## 📥 Dataset

Dataset yang digunakan adalah **Nottingham Music Database** — koleksi lagu folk Inggris dan Amerika yang merupakan benchmark standar untuk penelitian music generation.

| Info | Detail |
|------|--------|
| **Nama** | Nottingham Music Database |
| **Sumber** | https://github.com/jukedeck/nottingham-dataset |
| **Lisensi** | Public Domain |
| **Jumlah file** | 1.034 file MIDI melody |
| **Genre** | Folk (jigs, reels, waltzes, hornpipes, dll.) |
| **Format** | `.mid` |

Dataset sudah tersedia di folder `data/nottingham-source/` dalam repository ini.

---

## 🔄 Urutan Penggunaan

### 1. Preprocessing Data

```bash
python src/preprocessing/midi_parser.py --input_dir data/nottingham-source/MIDI/melody --output_dir data/processed
```

**Output:** `data/processed/` berisi `train.npz`, `val.npz`, `test.npz`, `vocab.pkl`

### 2. Training Model

```bash
# Windows
python src/model/train.py --data_dir data/processed --epochs 100 --batch_size 64

# Linux/Mac
python src/model/train.py \
    --data_dir data/processed \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 256 \
    --num_layers 2 \
    --lr 0.001 \
    --seed 42
```

**Output:** `models/best_model.pt` + `outputs/logs/training_log.json`

### 3. Generate Melodi

```bash
# Windows
python src/inference/generate.py --model_path models/best_model.pt --length 200 --temperature 0.8

# Linux/Mac — dengan seed note (C4=60, E4=64, G4=67)
python src/inference/generate.py \
    --model_path models/best_model.pt \
    --seed_notes 60 64 67 \
    --length 200 \
    --temperature 0.8
```

**Output:** `outputs/midi/generated.mid` + piano roll di `outputs/plots/`

### 4. Evaluasi

```bash
python src/evaluation/metrics.py --midi_path outputs/midi/generated.mid --log_path outputs/logs/training_log.json --dataset_dir data/nottingham-source/MIDI/melody
```

**Output:** `outputs/logs/evaluation_report.json` + grafik di `outputs/plots/`

### 5. Jalankan Aplikasi

```bash
python app/app.py
```

Buka browser: **http://localhost:7860**

---

## 🏋️ Parameter Training

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--epochs` | 100 | Jumlah epoch training |
| `--batch_size` | 64 | Ukuran batch |
| `--lr` | 0.001 | Learning rate |
| `--hidden_size` | 256 | Ukuran hidden state LSTM |
| `--num_layers` | 2 | Jumlah layer LSTM |
| `--dropout` | 0.3 | Dropout probability |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed untuk reproducibility |

---

## 🎼 Parameter Generation

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--length` | 200 | Panjang melodi (jumlah note) |
| `--temperature` | 0.8 | Kreativitas (0.1=konservatif, 2.0=kreatif) |
| `--top_k` | 10 | Jumlah kandidat note teratas |
| `--seed_notes` | random | Pitch MIDI awal (contoh: 60 64 67) |
| `--tempo` | 120 | Tempo dalam BPM |

---

## 🖥️ Cara Penggunaan Aplikasi

1. Jalankan `python app/app.py` dan buka **http://localhost:7860**
2. Atur **panjang melodi** menggunakan slider (50–500 note)
3. Atur **temperature** (rendah=terstruktur, tinggi=kreatif)
4. Isi **seed pitch** secara manual atau biarkan kosong untuk random
5. Klik **"Generate Melodi"**
6. Lihat visualisasi piano roll dan **download file `.mid`**

---

## 📊 Evaluasi

Model dievaluasi menggunakan kombinasi metrik kuantitatif dan kualitatif:

### Metrik Kuantitatif

| Metrik | Keterangan |
|--------|------------|
| **Loss (Cross-Entropy)** | Diukur tiap epoch selama training |
| **Perplexity** | Seberapa "yakin" model memprediksi note berikutnya |
| **Pitch Diversity Score** | Unique pitch / possible pitch range (0–1) |
| **Repetition Score** | N-gram repetition ratio (0=tidak repetitif) |
| **KL Divergence** | Jarak distribusi pitch generated vs dataset |

### Evaluasi Kualitatif (Human Evaluation)

Melodi dinilai oleh pendengar menggunakan rubrik skala 1–5:

| Kriteria | Deskripsi |
|----------|-----------|
| **Koherensi** | Apakah melodi terdengar masuk akal dan tidak random? |
| **Musikalitas** | Apakah ada pola ritmis yang dapat dirasakan? |
| **Keragaman** | Apakah tiap generate menghasilkan melodi berbeda? |
| **Kemiripan Gaya** | Apakah terasa seperti musik folk? |

---

## 🔗 Link Penting

| Item | Link |
|------|------|
| 📁 Repository | https://github.com/ArifbillahKamil/GeMelan |
| 📦 Dataset | https://github.com/jukedeck/nottingham-dataset |
| 💾 Model Checkpoint | [akan diisi] |
| 🚀 Demo Aplikasi | [akan diisi] |
| 📄 Laporan | [akan diisi] |

---

## 📋 Catatan

- Model dijalankan secara lokal tanpa bergantung pada internet atau API eksternal
- Training dapat dilakukan di CPU maupun GPU — GPU mempercepat proses secara signifikan
- Keterbatasan resource (CPU-only untuk sebagian anggota) didokumentasikan di laporan
- Dataset Nottingham Music Database berstatus **Public Domain** — bebas digunakan untuk keperluan akademik
