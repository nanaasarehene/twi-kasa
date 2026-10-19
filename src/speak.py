"""Read Twi text aloud and save it as a .wav file.

Usage:
  python src/speak.py "Akwaaba, wo ho te sɛn?"
  python src/speak.py "Me din de Ama." --out ama.wav
  python src/speak.py --file data/external/unseen_sentences.txt --out-dir outputs/samples/kasa
  python src/speak.py "Akwaaba" --model facebook/mms-tts-aka      # the untouched base model

Text is cleaned with the same twi_text.for_model() used in training, then
spoken one sentence at a time with a short pause in between.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

from twi_text import for_model, split_sentences, clean

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "checkpoints/kasa-twi"


class Speaker:
    def __init__(self, model=DEFAULT_MODEL, seed=0, noise_scale=0.667, speaking_rate=1.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VitsModel.from_pretrained(str(model)).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model))
        self.sr = self.model.config.sampling_rate
        # VITS samples random noise when speaking; fixing the seed makes runs repeatable
        self.seed = seed
        self.model.noise_scale = noise_scale
        self.model.speaking_rate = speaking_rate

    @torch.inference_mode()
    def say(self, text, pause=0.25):
        pieces = []
        for sentence in split_sentences(clean(text)):
            ids = self.tokenizer(for_model(sentence), return_tensors="pt").to(self.device)
            if ids.input_ids.shape[1] < 2:
                continue
            torch.manual_seed(self.seed)
            pieces.append(self.model(**ids).waveform[0].cpu().numpy())
            pieces.append(np.zeros(int(pause * self.sr), dtype=np.float32))
        if not pieces:
            raise ValueError(f"nothing speakable in {text!r}")
        return np.concatenate(pieces[:-1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="Twi text to speak")
    p.add_argument("--file", type=Path, help="text file, one prompt per line")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model folder or Hugging Face id")
    p.add_argument("--out", type=Path, default=Path("speech.wav"))
    p.add_argument("--out-dir", type=Path, help="where to write one wav per line of --file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-scale", type=float, default=0.667)
    p.add_argument("--rate", type=float, default=1.0, help="speaking rate, 1.0 = normal")
    args = p.parse_args()

    speaker = Speaker(args.model, args.seed, args.noise_scale, args.rate)
    if args.file:
        out_dir = args.out_dir or Path("outputs/samples")
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [l.strip() for l in args.file.read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(out_dir / "prompts.tsv", "w", encoding="utf-8") as index:
            for i, line in enumerate(lines, 1):
                name = f"{i:02d}.wav"
                sf.write(out_dir / name, speaker.say(line), speaker.sr)
                index.write(f"{name}\t{line}\n")
                print(f"{name}  {line}")
    elif args.text:
        sf.write(args.out, speaker.say(args.text), speaker.sr)
        print(f"wrote {args.out}")
    else:
        p.error("give some text or --file")


if __name__ == "__main__":
    main()
