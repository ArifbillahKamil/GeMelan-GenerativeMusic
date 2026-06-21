"""
generate.py
===========
Script inference untuk menghasilkan melodi baru menggunakan
model MelodyLSTM yang sudah ditraining.

Strategi generation:
    - Temperature sampling: semakin tinggi temperature, semakin kreatif/acak
    - Top-k sampling: hanya ambil k token teratas sebelum sampling
    - Seed notes: bisa mulai dari note tertentu atau acak

Output:
    - File MIDI (.mid) yang bisa diputar di media player apapun

Penggunaan:
    python src/inference/generate.py
    python src/inference/generate.py --length 200 --temperature 0.8
    python src/inference/generate.py --seed_notes 60 64 67 --temperature 1.0
"""

import os
import sys
import argparse
import pickle
import random

import numpy as np
import torch
import pretty_midi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model.lstm import MelodyLSTM



# 1. LOAD MODEL & VOCAB


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Memuat model dari checkpoint.

    Args:
        checkpoint_path : path ke file .pt hasil training
        device          : cpu / cuda

    Returns:
        Tuple (model, model_config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint['model_config']

    model = MelodyLSTM(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[Generate] Model loaded dari '{checkpoint_path}'")
    print(f"  Epoch      : {checkpoint['epoch']}")
    print(f"  Val loss   : {checkpoint['val_loss']:.4f}")
    model.summary()

    return model, config


def load_vocab(vocab_path: str) -> dict:
    """Memuat vocabulary dari file pickle."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"[Generate] Vocab loaded: {vocab['vocab_size']} token")
    return vocab



# 2. SAMPLING STRATEGY


def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Temperature sampling untuk memilih token berikutnya.

    Temperature mengontrol "kreativitas" generasi:
        - temperature < 1.0 : lebih konservatif, melodi lebih repetitif
        - temperature = 1.0 : distribusi asli dari model
        - temperature > 1.0 : lebih kreatif/acak, melodi lebih bervariasi

    Args:
        logits      : raw logit dari model, shape (vocab_size,)
        temperature : nilai temperature

    Returns:
        Index token yang dipilih (integer)
    """
    # Bagi logit dengan temperature sebelum softmax
    scaled_logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def topk_sampling(logits: torch.Tensor, temperature: float = 1.0, k: int = 10) -> int:
    """
    Top-k sampling: hanya pertimbangkan k token dengan probabilitas tertinggi.

    Menghindari token dengan probabilitas sangat rendah dipilih,
    sambil tetap memberikan variasi output.

    Args:
        logits      : raw logit dari model, shape (vocab_size,)
        temperature : nilai temperature
        k           : jumlah top token yang dipertimbangkan

    Returns:
        Index token yang dipilih (integer)
    """
    scaled_logits = logits / max(temperature, 1e-8)

    # Ambil top-k nilai dan index-nya
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k=min(k, logits.size(-1)))

    # Softmax hanya pada top-k
    probs = torch.softmax(top_k_logits, dim=-1)

    # Sample dari distribusi top-k
    sampled = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[sampled].item()



# 3. GENERATION LOOP


def generate_sequence(
    model:       MelodyLSTM,
    vocab:       dict,
    seed_tokens: list[int],
    length:      int = 200,
    temperature: float = 0.8,
    top_k:       int = 10,
    device:      torch.device = torch.device('cpu'),
) -> list[int]:
    """
    Menghasilkan sequence token melodi baru.

    Proses:
        1. Mulai dari seed tokens (warm-up hidden state)
        2. Secara autoregressive generate token baru satu per satu
        3. Berhenti jika mencapai panjang target atau token <EOS>

    Args:
        model       : MelodyLSTM yang sudah diload
        vocab       : vocabulary dictionary
        seed_tokens : list token awal (warm-up)
        length      : jumlah note yang ingin digenerate
        temperature : kreativitas sampling
        top_k       : k untuk top-k sampling
        device      : cpu / cuda

    Returns:
        List token integer hasil generasi (tanpa special tokens)
    """
    EOS_IDX = vocab['special']['EOS']
    PAD_IDX = vocab['special']['PAD']
    SOS_IDX = vocab['special']['SOS']

    model.eval()

    with torch.no_grad():
        # Inisialisasi hidden state
        hidden = model.init_hidden(1, device)

        # Warm-up: proses seed tokens untuk bangun konteks
        generated = []
        input_token = torch.tensor([[SOS_IDX]], dtype=torch.long).to(device)
        _, hidden = model(input_token, hidden)

        for token in seed_tokens:
            input_token = torch.tensor([[token]], dtype=torch.long).to(device)
            _, hidden = model(input_token, hidden)
            generated.append(token)

        # Autoregressive generation
        current_token = seed_tokens[-1] if seed_tokens else SOS_IDX

        for _ in range(length):
            input_token = torch.tensor([[current_token]], dtype=torch.long).to(device)
            logits, hidden = model(input_token, hidden)

            # logits shape: (1, 1, vocab_size) -> (vocab_size,)
            next_logits = logits[0, 0, :]

            # Masking special tokens agar tidak digenerate
            next_logits[PAD_IDX] = float('-inf')
            next_logits[SOS_IDX] = float('-inf')

            # Pilih token berikutnya dengan top-k + temperature sampling
            next_token = topk_sampling(next_logits, temperature=temperature, k=top_k)

            # Berhenti jika EOS
            if next_token == EOS_IDX:
                break

            generated.append(next_token)
            current_token = next_token

    return generated


# ---------------------------------------------------------------------------
# 4. KONVERSI TOKEN -> MIDI
# ---------------------------------------------------------------------------

def tokens_to_midi(
    tokens:     list[int],
    vocab:      dict,
    output_path: str,
    tempo:      float = 120.0,
    instrument_program: int = 0,
) -> pretty_midi.PrettyMIDI:
    """
    Mengkonversi sequence token integer menjadi file MIDI.

    Args:
        tokens             : list token integer hasil generasi
        vocab              : vocabulary dictionary
        output_path        : path untuk menyimpan file .mid
        tempo              : tempo dalam BPM (default: 120)
        instrument_program : MIDI program number (0=Piano, 40=Violin, dll.)

    Returns:
        PrettyMIDI object
    """
    idx2token = vocab['idx2token']

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(
        program=instrument_program,
        name='GeMelan Melody'
    )

    current_time = 0.0
    note_count   = 0

    for token_idx in tokens:
        token = idx2token.get(token_idx)

        # Skip jika token tidak dikenal atau special token (string)
        if token is None or isinstance(token, str):
            continue

        pitch, duration = token

        # Validasi nilai pitch dan duration
        if not (0 <= pitch <= 127) or duration <= 0:
            continue

        note = pretty_midi.Note(
            velocity=90,
            pitch=int(pitch),
            start=current_time,
            end=current_time + duration,
        )
        instrument.notes.append(note)
        current_time += duration
        note_count += 1

    midi.instruments.append(instrument)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    midi.write(output_path)

    print(f"[Generate] MIDI disimpan: '{output_path}'")
    print(f"  Notes     : {note_count}")
    print(f"  Duration  : {current_time:.2f} detik ({current_time/60:.2f} menit)")
    print(f"  Tempo     : {tempo} BPM")

    return midi


# ---------------------------------------------------------------------------
# 5. VISUALISASI PIANO ROLL (OPSIONAL)
# ---------------------------------------------------------------------------

def save_piano_roll(midi_obj: pretty_midi.PrettyMIDI, output_path: str) -> None:
    """
    Menyimpan visualisasi piano roll dari MIDI sebagai gambar PNG.

    Piano roll menampilkan pitch (sumbu Y) vs waktu (sumbu X),
    memudahkan analisis visual pola melodi yang dihasilkan.

    Args:
        midi_obj    : PrettyMIDI object
        output_path : path untuk menyimpan gambar .png
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(14, 4))

        colors = ['#4C9BE8', '#E8714C', '#4CE87A']

        for i, instrument in enumerate(midi_obj.instruments):
            color = colors[i % len(colors)]
            for note in instrument.notes:
                ax.barh(
                    y=note.pitch,
                    width=note.end - note.start,
                    left=note.start,
                    height=0.8,
                    color=color,
                    alpha=0.8,
                )

        ax.set_xlabel('Waktu (detik)', fontsize=11)
        ax.set_ylabel('Pitch MIDI', fontsize=11)
        ax.set_title('GeMelan — Piano Roll Melodi yang Dihasilkan', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"[Generate] Piano roll disimpan: '{output_path}'")

    except ImportError:
        print("[Generate] matplotlib tidak tersedia, skip piano roll.")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def run_generation(args):
    """Menjalankan full generation pipeline."""

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cpu')

    print("=" * 55)
    print("  GeMelan - Melody Generation")
    print("=" * 55)

    # Load model & vocab
    model = load_model(args.model_path, device)
    model, _ = model if isinstance(model, tuple) else (model, {})
    vocab = load_vocab(args.vocab_path)

    token2idx = vocab['token2idx']
    SOS_IDX   = vocab['special']['SOS']

    # Siapkan seed tokens
    if args.seed_notes:
        # Cari token terdekat untuk setiap seed pitch
        seed_tokens = []
        for pitch in args.seed_notes:
            # Cari token dengan pitch ini dan duration 0.25 (quarter note)
            candidates = [
                (k, v) for k, v in token2idx.items()
                if isinstance(k, tuple) and k[0] == pitch
            ]
            if candidates:
                # Pilih duration 0.25 jika ada, atau yang pertama
                token = next(
                    (v for k, v in candidates if abs(k[1] - 0.25) < 0.01),
                    candidates[0][1]
                )
                seed_tokens.append(token)

        if not seed_tokens:
            print(f"[WARNING] Seed notes tidak ditemukan di vocab, gunakan random seed.")
            seed_tokens = [random.choice([
                v for k, v in token2idx.items() if isinstance(k, tuple)
            ])]
    else:
        # Random seed: pilih note acak dari vocab
        note_tokens = [v for k, v in token2idx.items() if isinstance(k, tuple)]
        seed_tokens = [random.choice(note_tokens)]

    print(f"\n[Generate] Parameter:")
    print(f"  Length      : {args.length} notes")
    print(f"  Temperature : {args.temperature}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Seed notes  : {args.seed_notes if args.seed_notes else 'random'}")
    print(f"  Tempo       : {args.tempo} BPM")

    # Generate sequence
    print(f"\n[Generate] Generating melody...")
    tokens = generate_sequence(
        model=model,
        vocab=vocab,
        seed_tokens=seed_tokens,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(f"  Generated   : {len(tokens)} tokens")

    # Konversi ke MIDI
    midi_obj = tokens_to_midi(
        tokens=tokens,
        vocab=vocab,
        output_path=args.output,
        tempo=args.tempo,
    )

    # Simpan piano roll
    plot_path = args.output.replace('.mid', '_piano_roll.png')
    plot_path = os.path.join('outputs/plots', os.path.basename(plot_path))
    save_piano_roll(midi_obj, plot_path)

    print("\n" + "=" * 55)
    print("  Generation selesai!")
    print(f"  MIDI output : {args.output}")
    print(f"  Piano roll  : {plot_path}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeMelan - Generate Melody')

    parser.add_argument('--model_path',  type=str,   default='models/best_model.pt',
                        help='Path ke checkpoint model')
    parser.add_argument('--vocab_path',  type=str,   default='data/processed/vocab.pkl',
                        help='Path ke file vocab.pkl')
    parser.add_argument('--output',      type=str,   default='outputs/midi/generated.mid',
                        help='Path output file MIDI')
    parser.add_argument('--length',      type=int,   default=200,
                        help='Jumlah note yang digenerate (default: 200)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature sampling (default: 0.8)')
    parser.add_argument('--top_k',       type=int,   default=10,
                        help='Top-k sampling (default: 10)')
    parser.add_argument('--seed_notes',  type=int,   nargs='+', default=None,
                        help='Seed pitch MIDI (contoh: --seed_notes 60 64 67)')
    parser.add_argument('--tempo',       type=float, default=120.0,
                        help='Tempo BPM (default: 120)')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    run_generation(args)