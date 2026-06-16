# 🎵 GeMelan — Indonesian Traditional Melody Generator

> Aplikasi Generative AI yang menghasilkan melodi musik tradisional Indonesia secara otomatis menggunakan model LSTM yang dibangun dari scratch dengan PyTorch.

---

## 👥 Anggota Kelompok

| Nama | NIM | Kontribusi |
|------|-----|------------|
| [Nama 1] | [NIM 1] | Preprocessing & Evaluasi |
| [Nama 2] | [NIM 2] | Arsitektur Model & Training |
| [Nama 3] | [NIM 3] | Interface & Dokumentasi |

**Kelas:** [Kelas]  
**Mata Kuliah:** Generative AI  
**Link Repository:** [https://github.com/username/GeMelan](https://github.com/username/GeMelan)  
**Link Demo:** [akan diisi setelah deploy]  

---

## 📌 Deskripsi Proyek

**GeMelan** adalah sistem generatif berbasis **LSTM (Long Short-Term Memory)** yang dibangun dari scratch menggunakan PyTorch. Model belajar pola melodi dari dataset MIDI musik tradisional Indonesia, kemudian menghasilkan melodi baru yang orisinal dalam format `.mid` yang dapat diputar dan didownload.

### Fitur Utama
- 🎼 Generate melodi baru dari seed note atau secara acak
- 🎛️ Kontrol parameter: panjang melodi & temperature (kreativitas)
- 💾 Download hasil generate dalam format MIDI (`.mid`)
- 📊 Visualisasi piano roll dari melodi yang dihasilkan
- 🖥️ Antarmuka web interaktif berbasis Gradio

> ✅ **Tidak menggunakan provider LLM/API tertutup** (OpenAI, Anthropic, Google Gemini, dsb.)  
> Model dibangun sepenuhnya dari scratch menggunakan PyTorch tanpa bergantung pada layanan eksternal berbayar.

---

## 🗂️ Struktur Folder

```
GeMelan/
├── data/                   # Dataset MIDI (lihat instruksi unduh di bawah)
├── src/
│   ├── model/              # Arsitektur LSTM & training loop
│   │   ├── lstm.py         # Definisi model LSTM
│   │   └── train.py        # Script training
│   ├── preprocessing/      # Parsing & tokenisasi MIDI
│   │   ├── midi_parser.py  # Parser file MIDI ke sequence note
│   │   └── tokenizer.py    # Konversi note ke token integer
│   ├── evaluation/         # Metrik evaluasi
│   │   └── metrics.py      # Perplexity, diversity, repetition score
│   └── inference/          # Generate melodi baru
│       └── generate.py     # Script inference & sampling
├── notebooks/              # Eksplorasi & eksperimen Jupyter
├── models/                 # Checkpoint model hasil training
├── outputs/
│   ├── midi/               # Hasil generate melodi (.mid)
│   ├── plots/              # Grafik loss, distribusi pitch
│   └── logs/               # Log training
├── app/
│   └── app.py              # Gradio interface
├── requirements.txt        # Daftar dependencies
├── .env.example            # Contoh environment variable
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
git clone https://github.com/[username]/GeMelan.git
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

Dataset yang digunakan adalah kumpulan file MIDI musik tradisional Indonesia.

| Info | Detail |
|------|--------|
| **Sumber** | [akan diisi setelah dataset fix] |
| **Lisensi** | [akan diisi] |
| **Jumlah file** | [akan diisi] |
| **Format** | `.mid` / `.midi` |

### Cara Menyiapkan Dataset

```bash
# Unduh dataset dari link di atas, lalu letakkan di folder data/
# Struktur yang diharapkan:
data/
└── raw/
    ├── lagu1.mid
    ├── lagu2.mid
    └── ...
```

---

## 🏋️ Training Model

```bash
# Jalankan preprocessing terlebih dahulu
python src/preprocessing/midi_parser.py --data_dir data/raw --output_dir data/processed

# Mulai training
python src/model/train.py \
  --data_dir data/processed \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --hidden_size 256 \
  --num_layers 2 \
  --seed 42 \
  --checkpoint_dir models/
```

### Parameter Training

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--epochs` | 100 | Jumlah epoch training |
| `--batch_size` | 32 | Ukuran batch |
| `--lr` | 0.001 | Learning rate |
| `--hidden_size` | 256 | Ukuran hidden state LSTM |
| `--num_layers` | 2 | Jumlah layer LSTM |
| `--seed` | 42 | Random seed untuk reproducibility |

---

## 🎼 Generate Melodi (Inference)

```bash
# Generate melodi baru dengan parameter default
python src/inference/generate.py \
  --model_path models/best_model.pt \
  --length 200 \
  --temperature 0.8 \
  --output outputs/midi/result.mid

# Generate dengan seed note tertentu (pitch MIDI 0-127)
python src/inference/generate.py \
  --model_path models/best_model.pt \
  --seed_notes 60 64 67 \
  --length 200 \
  --temperature 0.8 \
  --output outputs/midi/result.mid
```

### Parameter Generation

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--length` | 200 | Panjang melodi (jumlah note) |
| `--temperature` | 0.8 | Kreativitas (0.1=konservatif, 1.5=kreatif) |
| `--seed_notes` | random | Note awal sebagai pemicu generate |

---

## 🖥️ Menjalankan Aplikasi

```bash
python app/app.py
```

Buka browser dan akses: **http://localhost:7860**

### Cara Penggunaan Aplikasi
1. Atur panjang melodi yang diinginkan menggunakan slider
2. Atur nilai temperature (semakin tinggi = semakin kreatif/acak)
3. Isi seed note secara manual atau biarkan kosong untuk generate acak
4. Klik tombol **"Generate Melodi"**
5. Dengarkan preview melodi dan download file `.mid`

---

## 📊 Evaluasi

Model dievaluasi menggunakan kombinasi metrik kuantitatif dan kualitatif:

### Metrik Kuantitatif
| Metrik | Keterangan |
|--------|------------|
| **Loss (Cross-Entropy)** | Diukur tiap epoch selama training |
| **Perplexity** | Mengukur seberapa "yakin" model memprediksi note berikutnya |
| **Pitch Diversity Score** | Jumlah unique pitch dalam output generate |
| **Repetition Score** | Seberapa sering pola note berulang (n-gram repetition) |

### Evaluasi Kualitatif (Human Evaluation)
Melodi hasil generate dinilai oleh pendengar menggunakan rubrik skala 1–5:
| Kriteria | Deskripsi |
|----------|-----------|
| **Koherensi** | Apakah melodi terdengar masuk akal dan tidak random? |
| **Musikalitas** | Apakah ada pola ritmis yang dapat dirasakan? |
| **Keragaman** | Apakah tiap generate menghasilkan melodi yang berbeda? |
| **Kemiripan Gaya** | Apakah terasa seperti musik tradisional Indonesia? |

---

## 🔗 Link Penting

| Item | Link |
|------|------|
| 📁 Dataset | [akan diisi] |
| 💾 Model Checkpoint | [akan diisi] |
| 🚀 Demo Aplikasi | [akan diisi] |
| 📄 Laporan | [akan diisi] |

---

## 📋 Catatan

- Model dijalankan secara lokal tanpa bergantung pada internet atau API eksternal
- Training dapat dilakukan di CPU, namun GPU akan mempercepat proses secara signifikan
- Keterbatasan resource (CPU-only) didokumentasikan di laporan beserta dampaknya terhadap kualitas output
