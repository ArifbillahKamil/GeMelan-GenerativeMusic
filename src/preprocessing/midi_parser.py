"""
midi_parser.py
==============
Preprocessing pipeline untuk Nottingham Music Database.

Alur:
    data/raw/*.mid
        -> parse note (pitch, duration)
        -> tokenisasi (pitch_duration -> integer token)
        -> windowing (fixed-length sequences)
        -> simpan ke data/processed/

Penggunaan:
    python src/preprocessing/midi_parser.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --seq_len 64 \
        --seed 42
"""

import os
import json
import argparse
import random
import pickle
from collections import Counter

import numpy as np
import pretty_midi
from tqdm import tqdm


# 1. PARSE MIDI -> LIST OF (pitch, duration) TUPLES

def parse_midi_file(filepath: str) -> list[tuple[int, float]]:
    """
    Membaca satu file MIDI dan mengembalikan sequence note.

    Setiap note direpresentasikan sebagai tuple (pitch, duration) di mana:
        - pitch    : integer MIDI pitch (0-127)
        - duration : durasi note dalam detik, dibulatkan ke grid terdekat

    Args:
        filepath: Path ke file .mid

    Returns:
        List of (pitch, duration) tuples, diurutkan berdasarkan waktu mulai.
    """
    midi = pretty_midi.PrettyMIDI(filepath)
    notes = []

    for instrument in midi.instruments:
        # Skip percussion track
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            duration = note.end - note.start
            notes.append((note.start, note.pitch, duration))

    # Urutkan berdasarkan waktu mulai
    notes.sort(key=lambda x: x[0])

    # Kembalikan hanya (pitch, duration) tanpa timestamp
    return [(pitch, duration) for (_, pitch, duration) in notes]


def quantize_duration(duration: float, grid: list[float]) -> float:
    """
    Membulatkan durasi ke nilai grid terdekat.

    Nottingham dataset hanya punya 13 unique duration,
    quantisasi memastikan konsistensi tokenisasi.

    Args:
        duration : durasi asli dalam detik
        grid     : list nilai durasi yang diperbolehkan

    Returns:
        Nilai grid yang paling dekat dengan duration.
    """
    return min(grid, key=lambda g: abs(g - duration))


def parse_all_midi(input_dir: str, duration_grid: list[float]) -> list[list[tuple[int, float]]]:
    """
    Memproses semua file MIDI dalam satu folder.

    Args:
        input_dir     : folder berisi file .mid
        duration_grid : list durasi valid untuk quantisasi

    Returns:
        List of songs, masing-masing berupa list of (pitch, duration) tuples.
    """
    midi_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.mid') or f.endswith('.midi')
    ])

    print(f"[Parser] Ditemukan {len(midi_files)} file MIDI di '{input_dir}'")

    songs = []
    skipped = 0

    for filepath in tqdm(midi_files, desc="Parsing MIDI"):
        try:
            notes = parse_midi_file(filepath)
            if len(notes) < 10:
                # Skip file yang terlalu pendek
                skipped += 1
                continue
            # Quantisasi duration
            notes = [(pitch, quantize_duration(dur, duration_grid)) for pitch, dur in notes]
            songs.append(notes)
        except Exception as e:
            print(f"  [WARNING] Skip {os.path.basename(filepath)}: {e}")
            skipped += 1

    print(f"[Parser] Berhasil: {len(songs)} lagu | Dilewati: {skipped}")
    return songs


# 2. TOKENISASI -> KONVERSI (pitch, duration) KE INTEGER TOKEN

def build_vocabulary(songs: list[list[tuple[int, float]]]) -> dict:
    """
    Membangun vocabulary dari seluruh dataset.

    Setiap pasangan (pitch, duration) unik menjadi satu token integer.
    Ditambahkan token khusus:
        <PAD> : padding untuk sequence pendek
        <SOS> : start of sequence
        <EOS> : end of sequence

    Args:
        songs: List of songs dari parse_all_midi()

    Returns:
        Dictionary berisi:
            'token2idx' : dict mapping (pitch, duration) -> integer
            'idx2token' : dict mapping integer -> (pitch, duration)
            'vocab_size': jumlah token unik termasuk special tokens
            'special'   : dict mapping nama special token -> integer
    """
    # Hitung frekuensi setiap (pitch, duration)
    counter = Counter()
    for song in songs:
        counter.update(song)

    # Urutkan berdasarkan frekuensi descending untuk konsistensi
    unique_tokens = sorted(counter.keys(), key=lambda x: -counter[x])

    # Special tokens
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    OFFSET = 3  # Token biasa dimulai dari index 3

    token2idx = {
        '<PAD>': PAD_IDX,
        '<SOS>': SOS_IDX,
        '<EOS>': EOS_IDX,
    }
    idx2token = {
        PAD_IDX: '<PAD>',
        SOS_IDX: '<SOS>',
        EOS_IDX: '<EOS>',
    }

    for i, token in enumerate(unique_tokens):
        idx = i + OFFSET
        token2idx[token] = idx
        idx2token[idx] = token

    vocab_size = len(token2idx)

    print(f"[Tokenizer] Vocabulary size: {vocab_size} token")
    print(f"  Special tokens: <PAD>={PAD_IDX}, <SOS>={SOS_IDX}, <EOS>={EOS_IDX}")
    print(f"  Note tokens: {OFFSET} - {vocab_size - 1}")

    return {
        'token2idx': token2idx,
        'idx2token': idx2token,
        'vocab_size': vocab_size,
        'special': {
            'PAD': PAD_IDX,
            'SOS': SOS_IDX,
            'EOS': EOS_IDX,
        }
    }


def encode_songs(songs: list[list[tuple[int, float]]], vocab: dict) -> list[list[int]]:
    """
    Mengkonversi semua lagu dari (pitch, duration) ke sequence integer token.

    Args:
        songs : output dari parse_all_midi()
        vocab : output dari build_vocabulary()

    Returns:
        List of encoded songs (list of integer tokens).
        Setiap song diawali <SOS> dan diakhiri <EOS>.
    """
    token2idx = vocab['token2idx']
    SOS = vocab['special']['SOS']
    EOS = vocab['special']['EOS']

    encoded = []
    unknown_count = 0

    for song in songs:
        tokens = [SOS]
        for note in song:
            if note in token2idx:
                tokens.append(token2idx[note])
            else:
                unknown_count += 1
        tokens.append(EOS)
        encoded.append(tokens)

    if unknown_count > 0:
        print(f"  [WARNING] {unknown_count} note tidak ada di vocabulary (dilewati)")

    total_tokens = sum(len(s) for s in encoded)
    print(f"[Tokenizer] Encoded {len(encoded)} lagu | Total tokens: {total_tokens}")
    return encoded


# 3. WINDOWING -> POTONG SEQUENCE JADI FIXED-LENGTH INPUT-TARGET PAIRS

def create_windows(
    encoded_songs: list[list[int]],
    seq_len: int,
    stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memotong sequence panjang menjadi pasangan (input, target) fixed-length.

    Untuk setiap window:
        input  = tokens[i : i + seq_len]
        target = tokens[i+1 : i + seq_len + 1]  (geser 1 langkah)

    Ini adalah formulation next-note prediction standar untuk language model.

    Args:
        encoded_songs : output dari encode_songs()
        seq_len       : panjang sequence input (jumlah token per window)
        stride        : langkah antar window (1 = overlap penuh, seq_len = no overlap)

    Returns:
        Tuple (X, y) di mana:
            X : array shape (num_windows, seq_len) — input sequences
            y : array shape (num_windows, seq_len) — target sequences
    """
    X, y = [], []

    for song in encoded_songs:
        # Minimal harus ada seq_len + 1 token
        if len(song) < seq_len + 1:
            continue

        for i in range(0, len(song) - seq_len, stride):
            input_seq  = song[i : i + seq_len]
            target_seq = song[i + 1 : i + seq_len + 1]
            X.append(input_seq)
            y.append(target_seq)

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    print(f"[Windowing] seq_len={seq_len}, stride={stride}")
    print(f"  Total windows: {len(X)}")
    print(f"  X shape: {X.shape} | y shape: {y.shape}")
    return X, y



# 4. TRAIN / VALIDATION / TEST SPLIT

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Membagi dataset menjadi train, validation, dan test set.

    Rasio default: 80% train / 10% val / 10% test

    Args:
        X, y        : output dari create_windows()
        train_ratio : proporsi data training
        val_ratio   : proporsi data validasi
        seed        : random seed untuk reproducibility

    Returns:
        Dictionary berisi X_train, y_train, X_val, y_val, X_test, y_test
    """
    assert train_ratio + val_ratio < 1.0, "train + val harus < 1.0"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_train = int(len(X) * train_ratio)
    n_val   = int(len(X) * val_ratio)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    splits = {
        'X_train': X[train_idx], 'y_train': y[train_idx],
        'X_val':   X[val_idx],   'y_val':   y[val_idx],
        'X_test':  X[test_idx],  'y_test':  y[test_idx],
    }

    print(f"[Split] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return splits



# 5. SIMPAN HASIL PREPROCESSING

def save_processed_data(splits: dict, vocab: dict, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Simpan split data
    np.savez(os.path.join(output_dir, 'train.npz'), X=splits['X_train'], y=splits['y_train'])
    np.savez(os.path.join(output_dir, 'val.npz'),   X=splits['X_val'],   y=splits['y_val'])
    np.savez(os.path.join(output_dir, 'test.npz'),  X=splits['X_test'],  y=splits['y_test'])

    # Simpan vocab sebagai pickle (karena key-nya tuple, tidak bisa JSON langsung)
    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    # Simpan metadata vocab sebagai JSON (untuk referensi manusia)
    vocab_meta = {
        'vocab_size': vocab['vocab_size'],
        'special_tokens': vocab['special'],
        'num_note_tokens': vocab['vocab_size'] - 3,
    }
    with open(os.path.join(output_dir, 'vocab_meta.json'), 'w') as f:
        json.dump(vocab_meta, f, indent=2)

    print(f"[Save] Hasil preprocessing disimpan di '{output_dir}':")
    print(f"  train.npz : {splits['X_train'].shape}")
    print(f"  val.npz   : {splits['X_val'].shape}")
    print(f"  test.npz  : {splits['X_test'].shape}")
    print(f"  vocab.pkl : {vocab['vocab_size']} token")


# 6. MAIN PIPELINE

# Duration grid dari hasil eksplorasi dataset Nottingham
# (13 nilai unik yang ditemukan di seluruh 1034 file)
DURATION_GRID = [0.062, 0.125, 0.167, 0.25, 0.333, 0.375,
                 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def run_preprocessing(
    input_dir:  str,
    output_dir: str,
    seq_len:    int = 64,
    stride:     int = 1,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int = 42
) -> None:

    print("=" * 55)
    print("  GeMelan — MIDI Preprocessing Pipeline")
    print("=" * 55)

    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Parse MIDI
    print("\n[Step 1/4] Parsing MIDI files...")
    songs = parse_all_midi(input_dir, DURATION_GRID)

    # Step 2: Bangun vocabulary & encode
    print("\n[Step 2/4] Membangun vocabulary & tokenisasi...")
    vocab = build_vocabulary(songs)
    encoded_songs = encode_songs(songs, vocab)

    # Step 3: Windowing
    print("\n[Step 3/4] Membuat windows (input-target pairs)...")
    X, y = create_windows(encoded_songs, seq_len=seq_len, stride=stride)

    # Step 4: Split & simpan
    print("\n[Step 4/4] Split dataset & menyimpan hasil...")
    splits = split_dataset(X, y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    save_processed_data(splits, vocab, output_dir)

    print("\n" + "=" * 55)
    print("  Preprocessing selesai!")
    print(f"  Vocab size  : {vocab['vocab_size']}")
    print(f"  Total windows: {len(X)}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 55)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GeMelan MIDI Preprocessing Pipeline'
    )
    parser.add_argument('--input_dir',   type=str,   default='data/raw',
                        help='Folder berisi file MIDI mentah')
    parser.add_argument('--output_dir',  type=str,   default='data/processed',
                        help='Folder output hasil preprocessing')
    parser.add_argument('--seq_len',     type=int,   default=64,
                        help='Panjang sequence window (default: 64)')
    parser.add_argument('--stride',      type=int,   default=1,
                        help='Stride antar window (default: 1)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proporsi data training (default: 0.8)')
    parser.add_argument('--val_ratio',   type=float, default=0.1,
                        help='Proporsi data validasi (default: 0.1)')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    run_preprocessing(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        seq_len     = args.seq_len,
        stride      = args.stride,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )