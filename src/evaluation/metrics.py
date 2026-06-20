"""
metrics.py
==========
Modul evaluasi untuk GeMelan melody generator.

Metrik yang tersedia:
    1. Perplexity          - dari training log
    2. Pitch Diversity     - jumlah unique pitch dalam output
    3. Repetition Score    - seberapa sering pola note berulang
    4. Pitch Distribution  - distribusi pitch vs dataset asli
    5. Human Eval Rubric   - template penilaian kualitatif

Penggunaan:
    python src/evaluation/metrics.py \
        --midi_path outputs/midi/generated.mid \
        --dataset_dir data/raw
"""

import os
import sys
import json
import argparse
import pickle
from collections import Counter

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# 1. PERPLEXITY (dari training log)
# ---------------------------------------------------------------------------

def load_perplexity_from_log(log_path: str) -> dict:
    """
    Membaca perplexity dari file log training.

    Perplexity = exp(loss) — mengukur seberapa "yakin" model
    memprediksi note berikutnya. Semakin rendah = semakin baik.

    Args:
        log_path : path ke training_log.json

    Returns:
        Dictionary berisi best, final, dan history perplexity
    """
    if not os.path.exists(log_path):
        print(f"[Metrics] Log tidak ditemukan: '{log_path}'")
        return {}

    with open(log_path, 'r') as f:
        log = json.load(f)

    perplexities = [entry['perplexity'] for entry in log]
    val_losses   = [entry['val_loss']   for entry in log]

    result = {
        'best_perplexity':  round(min(perplexities), 4),
        'final_perplexity': round(perplexities[-1], 4),
        'best_val_loss':    round(min(val_losses), 6),
        'total_epochs':     len(log),
        'history':          perplexities,
    }

    print(f"[Metrics] Perplexity:")
    print(f"  Best      : {result['best_perplexity']}")
    print(f"  Final     : {result['final_perplexity']}")
    print(f"  Epochs    : {result['total_epochs']}")

    return result


# ---------------------------------------------------------------------------
# 2. EKSTRAK NOTE DARI MIDI
# ---------------------------------------------------------------------------

def extract_notes_from_midi(midi_path: str) -> list[tuple[int, float]]:
    """
    Mengekstrak sequence (pitch, duration) dari file MIDI.

    Args:
        midi_path : path ke file .mid

    Returns:
        List of (pitch, duration) tuples
    """
    midi  = pretty_midi.PrettyMIDI(midi_path)
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch    = note.pitch
            duration = round(note.end - note.start, 3)
            notes.append((pitch, duration))

    return notes


def extract_notes_from_dataset(dataset_dir: str, max_files: int = 100) -> list[tuple[int, float]]:
    """
    Mengekstrak semua note dari dataset untuk perbandingan distribusi.

    Args:
        dataset_dir : folder berisi file MIDI dataset
        max_files   : maksimal file yang diproses (untuk efisiensi)

    Returns:
        List of (pitch, duration) tuples dari seluruh dataset
    """
    midi_files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith('.mid') or f.endswith('.midi')
    ])[:max_files]

    all_notes = []
    for filepath in midi_files:
        try:
            all_notes.extend(extract_notes_from_midi(filepath))
        except Exception:
            pass

    return all_notes


# ---------------------------------------------------------------------------
# 3. PITCH DIVERSITY SCORE
# ---------------------------------------------------------------------------

def pitch_diversity_score(notes: list[tuple[int, float]]) -> dict:
    """
    Mengukur keragaman pitch dalam melodi yang dihasilkan.

    Skor tinggi = melodi lebih bervariasi (banyak pitch berbeda).
    Skor rendah = melodi monoton (sedikit variasi pitch).

    Normalized score: unique_pitches / total_possible_pitches (0-1)

    Args:
        notes : list of (pitch, duration) tuples

    Returns:
        Dictionary berisi statistik diversity
    """
    if not notes:
        return {}

    pitches        = [n[0] for n in notes]
    unique_pitches = set(pitches)
    pitch_counts   = Counter(pitches)

    # Normalized diversity (0-1)
    # Range pitch di Nottingham: 55-88 = 34 possible pitches
    possible_range  = max(pitches) - min(pitches) + 1
    diversity_score = len(unique_pitches) / possible_range if possible_range > 0 else 0

    result = {
        'total_notes':     len(notes),
        'unique_pitches':  len(unique_pitches),
        'pitch_range':     f"{min(pitches)}-{max(pitches)}",
        'diversity_score': round(diversity_score, 4),
        'most_common_pitch': pitch_counts.most_common(3),
    }

    print(f"[Metrics] Pitch Diversity:")
    print(f"  Total notes    : {result['total_notes']}")
    print(f"  Unique pitches : {result['unique_pitches']}")
    print(f"  Pitch range    : {result['pitch_range']}")
    print(f"  Diversity score: {result['diversity_score']:.4f} (0=monoton, 1=sangat variatif)")

    return result


# ---------------------------------------------------------------------------
# 4. REPETITION SCORE
# ---------------------------------------------------------------------------

def repetition_score(notes: list[tuple[int, float]], n: int = 4) -> dict:
    """
    Mengukur seberapa sering pola note berulang (n-gram repetition).

    Mirip dengan self-BLEU — skor tinggi berarti melodi terlalu repetitif.
    Melodi yang baik punya sedikit repetisi tapi tetap koheren.

    Args:
        notes : list of (pitch, duration) tuples
        n     : panjang n-gram yang dicek (default: 4 note)

    Returns:
        Dictionary berisi statistik repetisi
    """
    if len(notes) < n:
        return {}

    pitches = [note[0] for note in notes]

    # Buat semua n-gram
    ngrams = [tuple(pitches[i:i+n]) for i in range(len(pitches) - n + 1)]
    total_ngrams  = len(ngrams)
    unique_ngrams = len(set(ngrams))

    # Repetition ratio: 1 - (unique/total)
    # Nilai mendekati 0 = jarang berulang (baik)
    # Nilai mendekati 1 = sangat repetitif (kurang baik)
    repetition_ratio = 1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0

    # Top repeated patterns
    ngram_counts = Counter(ngrams)
    top_repeated = ngram_counts.most_common(3)

    result = {
        'n':                n,
        'total_ngrams':     total_ngrams,
        'unique_ngrams':    unique_ngrams,
        'repetition_ratio': round(repetition_ratio, 4),
        'top_repeated':     [(list(k), v) for k, v in top_repeated],
    }

    print(f"[Metrics] Repetition Score ({n}-gram):")
    print(f"  Total n-grams  : {total_ngrams}")
    print(f"  Unique n-grams : {unique_ngrams}")
    print(f"  Repetition ratio: {repetition_ratio:.4f} (0=tidak repetitif, 1=sangat repetitif)")

    return result


# ---------------------------------------------------------------------------
# 5. PITCH DISTRIBUTION COMPARISON
# ---------------------------------------------------------------------------

def pitch_distribution_comparison(
    generated_notes: list[tuple[int, float]],
    dataset_notes:   list[tuple[int, float]],
) -> dict:
    """
    Membandingkan distribusi pitch output model vs dataset asli.

    Menggunakan KL Divergence sebagai metrik jarak distribusi.
    KL Divergence mendekati 0 = distribusi model mirip dataset (baik).

    Args:
        generated_notes : note dari hasil generasi model
        dataset_notes   : note dari dataset asli (sample)

    Returns:
        Dictionary berisi hasil perbandingan distribusi
    """
    gen_pitches  = [n[0] for n in generated_notes]
    data_pitches = [n[0] for n in dataset_notes]

    # Tentukan range pitch gabungan
    all_pitches = gen_pitches + data_pitches
    min_p, max_p = min(all_pitches), max(all_pitches)
    pitch_range  = list(range(min_p, max_p + 1))

    # Hitung distribusi (normalized)
    gen_counts  = Counter(gen_pitches)
    data_counts = Counter(data_pitches)

    epsilon = 1e-10  # Hindari log(0)

    gen_dist  = np.array([gen_counts.get(p, 0)  + epsilon for p in pitch_range], dtype=float)
    data_dist = np.array([data_counts.get(p, 0) + epsilon for p in pitch_range], dtype=float)

    gen_dist  /= gen_dist.sum()
    data_dist /= data_dist.sum()

    # KL Divergence: KL(generated || dataset)
    kl_div = float(np.sum(gen_dist * np.log(gen_dist / data_dist)))

    result = {
        'kl_divergence': round(kl_div, 6),
        'pitch_range':   f"{min_p}-{max_p}",
        'gen_unique':    len(set(gen_pitches)),
        'data_unique':   len(set(data_pitches)),
        'interpretation': 'Sangat mirip dataset' if kl_div < 0.1
                          else 'Mirip dataset' if kl_div < 0.5
                          else 'Cukup berbeda dari dataset' if kl_div < 1.0
                          else 'Sangat berbeda dari dataset',
    }

    print(f"[Metrics] Pitch Distribution Comparison:")
    print(f"  KL Divergence  : {result['kl_divergence']:.6f}")
    print(f"  Interpretasi   : {result['interpretation']}")

    return result


# ---------------------------------------------------------------------------
# 6. SIMPAN HASIL EVALUASI
# ---------------------------------------------------------------------------

def save_evaluation_report(results: dict, output_path: str) -> None:
    """
    Menyimpan seluruh hasil evaluasi ke file JSON.

    Args:
        results     : dictionary berisi semua hasil metrik
        output_path : path untuk menyimpan laporan
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[Metrics] Laporan evaluasi disimpan: '{output_path}'")


# ---------------------------------------------------------------------------
# 7. VISUALISASI METRIK
# ---------------------------------------------------------------------------

def plot_training_curve(log_path: str, output_path: str) -> None:
    """
    Membuat grafik loss dan perplexity selama training.

    Args:
        log_path    : path ke training_log.json
        output_path : path untuk menyimpan grafik .png
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not os.path.exists(log_path):
            print(f"[Metrics] Log tidak ditemukan, skip plot training curve.")
            return

        with open(log_path, 'r') as f:
            log = json.load(f)

        epochs      = [e['epoch']      for e in log]
        train_loss  = [e['train_loss'] for e in log]
        val_loss    = [e['val_loss']   for e in log]
        perplexity  = [e['perplexity'] for e in log]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        # Plot Loss
        ax1.plot(epochs, train_loss, label='Train Loss', color='#4C9BE8', linewidth=2)
        ax1.plot(epochs, val_loss,   label='Val Loss',   color='#E8714C', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot Perplexity
        ax2.plot(epochs, perplexity, color='#4CE87A', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Validation Perplexity')
        ax2.grid(alpha=0.3)

        plt.suptitle('GeMelan — Training Metrics', fontsize=13, fontweight='bold')
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"[Metrics] Training curve disimpan: '{output_path}'")

    except ImportError:
        print("[Metrics] matplotlib tidak tersedia, skip plot.")


def plot_pitch_distribution(
    generated_notes: list[tuple[int, float]],
    dataset_notes:   list[tuple[int, float]],
    output_path:     str,
) -> None:
    """
    Membuat grafik perbandingan distribusi pitch generated vs dataset.

    Args:
        generated_notes : note dari hasil generasi
        dataset_notes   : note dari dataset asli
        output_path     : path untuk menyimpan grafik .png
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gen_pitches  = [n[0] for n in generated_notes]
        data_pitches = [n[0] for n in dataset_notes]

        all_pitches = gen_pitches + data_pitches
        min_p, max_p = min(all_pitches), max(all_pitches)
        pitch_range  = list(range(min_p, max_p + 1))

        gen_counts  = Counter(gen_pitches)
        data_counts = Counter(data_pitches)

        gen_freq  = [gen_counts.get(p, 0)  / len(gen_pitches)  for p in pitch_range]
        data_freq = [data_counts.get(p, 0) / len(data_pitches) for p in pitch_range]

        fig, ax = plt.subplots(figsize=(13, 4))
        x = np.arange(len(pitch_range))
        width = 0.4

        ax.bar(x - width/2, data_freq, width, label='Dataset', color='#4C9BE8', alpha=0.7)
        ax.bar(x + width/2, gen_freq,  width, label='Generated', color='#E8714C', alpha=0.7)

        ax.set_xticks(x[::2])
        ax.set_xticklabels(pitch_range[::2], rotation=45, fontsize=8)
        ax.set_xlabel('MIDI Pitch')
        ax.set_ylabel('Frekuensi Relatif')
        ax.set_title('GeMelan — Pitch Distribution: Dataset vs Generated', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"[Metrics] Pitch distribution plot disimpan: '{output_path}'")

    except ImportError:
        print("[Metrics] matplotlib tidak tersedia, skip plot.")


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def run_evaluation(args):
    """Menjalankan full evaluasi pipeline."""

    print("=" * 55)
    print("  GeMelan - Evaluation Metrics")
    print("=" * 55)

    results = {}

    # 1. Perplexity dari training log
    print("\n[1/4] Perplexity dari training log...")
    perp = load_perplexity_from_log(args.log_path)
    if perp:
        results['perplexity'] = perp
        # Plot training curve
        plot_training_curve(
            args.log_path,
            'outputs/plots/training_curve.png'
        )

    # 2. Ekstrak note dari generated MIDI
    print(f"\n[2/4] Analisis generated MIDI: '{args.midi_path}'...")
    if not os.path.exists(args.midi_path):
        print(f"  [WARNING] File tidak ditemukan: '{args.midi_path}'")
        print("  Jalankan generate.py terlebih dahulu.")
        return

    generated_notes = extract_notes_from_midi(args.midi_path)
    print(f"  Extracted {len(generated_notes)} notes")

    # 3. Pitch diversity
    print("\n[3/4] Pitch diversity & repetition...")
    results['pitch_diversity'] = pitch_diversity_score(generated_notes)
    results['repetition']      = repetition_score(generated_notes, n=4)

    # 4. Pitch distribution vs dataset
    if args.dataset_dir and os.path.exists(args.dataset_dir):
        print(f"\n[4/4] Distribusi pitch vs dataset...")
        dataset_notes = extract_notes_from_dataset(args.dataset_dir, max_files=100)
        results['pitch_distribution'] = pitch_distribution_comparison(
            generated_notes, dataset_notes
        )
        plot_pitch_distribution(
            generated_notes, dataset_notes,
            'outputs/plots/pitch_distribution.png'
        )
    else:
        print(f"\n[4/4] Skip distribusi (--dataset_dir tidak ditemukan)")

    # Simpan laporan
    save_evaluation_report(results, args.output)

    # Ringkasan
    print("\n" + "=" * 55)
    print("  Ringkasan Evaluasi")
    print("=" * 55)
    if 'perplexity' in results:
        print(f"  Best Perplexity  : {results['perplexity'].get('best_perplexity', 'N/A')}")
    if 'pitch_diversity' in results:
        print(f"  Diversity Score  : {results['pitch_diversity'].get('diversity_score', 'N/A')}")
        print(f"  Unique Pitches   : {results['pitch_diversity'].get('unique_pitches', 'N/A')}")
    if 'repetition' in results:
        print(f"  Repetition Ratio : {results['repetition'].get('repetition_ratio', 'N/A')}")
    if 'pitch_distribution' in results:
        print(f"  KL Divergence    : {results['pitch_distribution'].get('kl_divergence', 'N/A')}")
        print(f"  Interpretasi     : {results['pitch_distribution'].get('interpretation', 'N/A')}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeMelan - Evaluation Metrics')

    parser.add_argument('--midi_path',   type=str, default='outputs/midi/generated.mid',
                        help='Path ke file MIDI hasil generasi')
    parser.add_argument('--log_path',    type=str, default='outputs/logs/training_log.json',
                        help='Path ke training log JSON')
    parser.add_argument('--dataset_dir', type=str, default='data/nottingham-source/MIDI/melody',
                        help='Folder dataset MIDI untuk perbandingan distribusi')
    parser.add_argument('--output',      type=str, default='outputs/logs/evaluation_report.json',
                        help='Path output laporan evaluasi JSON')

    args = parser.parse_args()
    run_evaluation(args)