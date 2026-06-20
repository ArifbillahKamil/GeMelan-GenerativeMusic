import os
import sys
import pickle
import random
import tempfile

import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model.lstm import MelodyLSTM
from src.inference.generate import (
    load_model, load_vocab, generate_sequence,
    tokens_to_midi, save_piano_roll
)
# KONFIGURASI PATH
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')
VOCAB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'vocab.pkl')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'midi')
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# LOAD MODEL & VOCAB SAAT STARTUP
print("=" * 55)
print("  GeMelan - Loading model...")
print("=" * 55)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL  = None
VOCAB  = None

def load_resources():
    """Load model dan vocab sekali saat aplikasi start."""
    global MODEL, VOCAB

    if not os.path.exists(MODEL_PATH):
        print(f"[App] ⚠️  Model belum tersedia di '{MODEL_PATH}'")
        print(f"[App] Jalankan training terlebih dahulu:")
        print(f"[App] python src/model/train.py")
        return False

    if not os.path.exists(VOCAB_PATH):
        print(f"[App] ⚠️  Vocab belum tersedia di '{VOCAB_PATH}'")
        print(f"[App] Jalankan preprocessing terlebih dahulu:")
        print(f"[App] python src/preprocessing/midi_parser.py")
        return False

    model_tuple = load_model(MODEL_PATH, DEVICE)
    MODEL = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
    VOCAB = load_vocab(VOCAB_PATH)

    print("[App] ✅ Model & vocab berhasil dimuat!")
    return True

MODEL_LOADED = load_resources()

# FUNGSI GENERATE (dipanggil oleh Gradio)
def generate_melody(
    length:      int,
    temperature: float,
    top_k:       int,
    seed_pitch:  str,
    tempo:       float,
    seed:        int,
) -> tuple:
    """
    Fungsi utama yang dipanggil saat user klik 'Generate'.

    Args:
        length      : jumlah note yang digenerate
        temperature : kreativitas (0.1-2.0)
        top_k       : top-k sampling
        seed_pitch  : pitch awal (string, bisa kosong)
        tempo       : BPM
        seed        : random seed

    Returns:
        Tuple (midi_path, piano_roll_path, info_text)
    """
    if not MODEL_LOADED or MODEL is None or VOCAB is None:
        return (
            None, None,
            "❌ Model belum tersedia. Jalankan training terlebih dahulu.\n"
            "Command: python src/model/train.py"
        )

    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)

    token2idx = VOCAB['token2idx']

    # Siapkan seed tokens
    seed_tokens = []
    if seed_pitch.strip():
        try:
            pitches = [int(p.strip()) for p in seed_pitch.split(',') if p.strip()]
            for pitch in pitches:
                candidates = [
                    (k, v) for k, v in token2idx.items()
                    if isinstance(k, tuple) and k[0] == pitch
                ]
                if candidates:
                    token = next(
                        (v for k, v in candidates if abs(k[1] - 0.25) < 0.01),
                        candidates[0][1]
                    )
                    seed_tokens.append(token)
        except ValueError:
            pass

    if not seed_tokens:
        note_tokens = [v for k, v in token2idx.items() if isinstance(k, tuple)]
        seed_tokens = [random.choice(note_tokens)]

    # Generate sequence
    tokens = generate_sequence(
        model=MODEL,
        vocab=VOCAB,
        seed_tokens=seed_tokens,
        length=length,
        temperature=temperature,
        top_k=top_k,
        device=DEVICE,
    )

    if not tokens:
        return None, None, "❌ Gagal generate melodi. Coba ubah parameter."

    # Simpan MIDI
    midi_filename = f"melody_t{temperature}_l{length}_s{seed}.mid"
    midi_path     = os.path.join(OUTPUT_DIR, midi_filename)
    midi_obj      = tokens_to_midi(tokens, VOCAB, midi_path, tempo=tempo)

    # Simpan piano roll
    plot_filename = midi_filename.replace('.mid', '_piano_roll.png')
    plot_path     = os.path.join(PLOTS_DIR, plot_filename)
    save_piano_roll(midi_obj, plot_path)

    # Hitung durasi
    duration_sec = sum(
        k[1] for idx in tokens
        if (k := VOCAB['idx2token'].get(idx)) is not None
        and isinstance(k, tuple)
    )

    info = (
        f"✅ Melodi berhasil digenerate!\n\n"
        f"📊 Statistik:\n"
        f"  • Notes generated : {len(tokens)}\n"
        f"  • Durasi          : {duration_sec:.1f} detik ({duration_sec/60:.2f} menit)\n"
        f"  • Tempo           : {tempo} BPM\n"
        f"  • Temperature     : {temperature}\n"
        f"  • Top-k           : {top_k}\n"
        f"  • Seed            : {seed}\n"
        f"  • Seed pitch      : {seed_pitch if seed_pitch.strip() else 'random'}\n\n"
        f"💡 Tips:\n"
        f"  • Temperature rendah (0.3-0.6) = melodi lebih terstruktur\n"
        f"  • Temperature tinggi (0.9-1.5) = melodi lebih kreatif/acak\n"
        f"  • Coba seed pitch: 67 (G4), 69 (A4), 71 (B4)"
    )

    return midi_path, plot_path, info


# GRADIO INTERFACE
def build_interface() -> gr.Blocks:
    """Membangun tampilan Gradio."""

    with gr.Blocks(title="🎵 GeMelan — Folk Melody Generator") as demo:

        # Header
        gr.Markdown("""
        # 🎵 GeMelan — Folk Melody Generator
        Aplikasi Generative AI yang menghasilkan melodi musik folk menggunakan **LSTM from scratch** (PyTorch).
        Model dilatih pada **Nottingham Music Database** (1.034 lagu folk, public domain).

        > ✅ Tidak menggunakan API LLM eksternal (OpenAI, Anthropic, Gemini, dsb.)
        """)

        with gr.Row():
            # Kolom kiri - Parameter
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parameter Generasi")

                length = gr.Slider(
                    minimum=50, maximum=500, value=200, step=10,
                    label="Panjang Melodi (jumlah note)",
                    info="Semakin panjang, semakin lama durasi melodi"
                )

                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                    label="Temperature (kreativitas)",
                    info="Rendah=terstruktur, Tinggi=kreatif/acak"
                )

                top_k = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Top-k Sampling",
                    info="Jumlah kandidat note teratas yang dipertimbangkan"
                )

                tempo = gr.Slider(
                    minimum=60, maximum=200, value=120, step=5,
                    label="Tempo (BPM)",
                    info="Kecepatan melodi dalam beats per minute"
                )

                seed_pitch = gr.Textbox(
                    label="Seed Pitch (opsional)",
                    placeholder="Contoh: 67, 69, 71",
                    info="Pitch MIDI awal (kosongkan untuk random). C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71"
                )

                seed = gr.Number(
                    value=42, label="Random Seed",
                    info="Ubah untuk mendapatkan melodi berbeda dengan parameter yang sama"
                )

                btn_generate = gr.Button(
                    "🎼 Generate Melodi", variant="primary", size="lg"
                )

            # Kolom kanan - Output
            with gr.Column(scale=1):
                gr.Markdown("### 🎵 Hasil Generasi")

                piano_roll = gr.Image(
                    label="Piano Roll Visualization",
                    type="filepath",
                )

                midi_output = gr.File(
                    label="Download File MIDI (.mid)",
                    file_types=['.mid'],
                )

                info_text = gr.Textbox(
                    label="Informasi Generasi",
                    lines=12,
                    interactive=False,
                )

        # Contoh preset
        gr.Markdown("### 🎛️ Preset Cepat")
        with gr.Row():
            gr.Examples(
                examples=[
                    [200, 0.6, 10, "67, 69, 71", 120, 42],
                    [150, 1.0, 15, "",           100, 7],
                    [300, 0.4, 5,  "60, 64, 67", 140, 123],
                    [200, 1.5, 20, "",           80,  99],
                ],
                inputs=[length, temperature, top_k, seed_pitch, tempo, seed],
                label="Klik untuk mengisi parameter"
            )

        # Footer
        gr.Markdown("""
        ---
        **GeMelan** | Tugas Besar Generative AI | Telkom University
        Model: LSTM from scratch (PyTorch) | Dataset: Nottingham Music Database (Public Domain)
        """)

        # Event handler
        btn_generate.click(
            fn=generate_melody,
            inputs=[length, temperature, top_k, seed_pitch, tempo, seed],
            outputs=[midi_output, piano_roll, info_text],
        )

    return demo

# ENTRY POINT
if __name__ == '__main__':
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )