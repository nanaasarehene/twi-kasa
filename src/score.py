"""Measure how understandable speech is, using an Akan speech recogniser.

The recogniser (Meta's mms-1b-all with its Akan adapter) transcribes each clip
and we compare that with the text that was meant to be spoken:

  CER, WER      character / word error rate (lower is better)
  ɛ/ɔ kept      of all ɛ and ɔ in the text, how many came back as ɛ/ɔ
                rather than e/o or something else

Ways to use it:
  python src/score.py data/processed/test                     # real recordings: the ceiling
  python src/score.py data/processed/test --model facebook/mms-tts-aka   # base model says the same text
  python src/score.py data/processed/test --model checkpoints/kasa-twi   # our model says the same text
  python src/score.py data/processed/test --model base=facebook/mms-tts-aka --model step-500=runs/...
                                                              # several models, recogniser loaded once
  python src/score.py outputs/samples/kasa                    # any folder made by speak.py

Results go to outputs/eval/<name>/ (one row per clip + a summary). With several
models, each summary is written as soon as that model is done.
Caveat: the recogniser makes its own mistakes, so even real speech won't score 0.
That's why the real-recording score is reported alongside.
"""
import argparse
import csv
import json
from pathlib import Path

import jiwer
import soundfile as sf
import torch
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC

from twi_text import for_model

ROOT = Path(__file__).resolve().parent.parent
ASR_MODEL = "facebook/mms-1b-all"
OPEN_VOWELS = {"ɛ": "e", "ɔ": "o"}


class Listener:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL, target_lang="aka")
        self.model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL, target_lang="aka",
                                                    ignore_mismatched_sizes=True).eval().to(self.device)
        self.sr = self.processor.feature_extractor.sampling_rate

    @torch.inference_mode()
    def transcribe(self, wave, sr):
        wave = torch.as_tensor(wave, dtype=torch.float32)
        if sr != self.sr:
            wave = torchaudio.functional.resample(wave, sr, self.sr)
        inputs = self.processor(wave.numpy(), sampling_rate=self.sr, return_tensors="pt").to(self.device)
        ids = self.model(**inputs).logits.argmax(-1)[0]
        return self.processor.decode(ids)


def open_vowel_counts(reference, hypothesis):
    """Count ɛ/ɔ in the reference that were recognised as themselves, as e/o, or lost."""
    counts = {"kept": 0, "flattened": 0, "other": 0}
    out = jiwer.process_characters(reference, hypothesis)
    ref, hyp = out.references[0], out.hypotheses[0]
    for chunk in out.alignments[0]:
        for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
            r = ref[chunk.ref_start_idx + k]
            if r not in OPEN_VOWELS:
                continue
            if chunk.type == "equal":
                counts["kept"] += 1
            elif chunk.type == "substitute" and hyp[chunk.hyp_start_idx + k] == OPEN_VOWELS[r]:
                counts["flattened"] += 1
            else:
                counts["other"] += 1
    return counts


def read_items(folder):
    """(name, text) pairs from a speak.py folder (prompts.tsv) or a data split (metadata.csv)."""
    if (folder / "prompts.tsv").exists():
        rows = [l.split("\t", 1) for l in (folder / "prompts.tsv").read_text(encoding="utf-8").splitlines() if l]
        return [(name, text) for name, text in rows]
    with open(folder / "metadata.csv", encoding="utf-8") as f:
        return [(r["file_name"], r["text"]) for r in csv.DictReader(f)]


def score(items, folder, out_dir, listener, speaker=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if speaker:
        (out_dir / "audio").mkdir(exist_ok=True)
    rows, refs, hyps = [], [], []
    vowels = {"kept": 0, "flattened": 0, "other": 0}
    for file_name, text in items:
        if speaker:
            wave, sr = speaker.say(text), speaker.sr
            sf.write(out_dir / "audio" / file_name, wave, sr)
        else:
            wave, sr = sf.read(folder / file_name, dtype="float32")
        reference = for_model(text)
        heard = for_model(listener.transcribe(wave, sr))
        refs.append(reference)
        hyps.append(heard)
        v = open_vowel_counts(reference, heard)
        for k in vowels:
            vowels[k] += v[k]
        rows.append({"file_name": file_name, "text": reference, "heard": heard,
                     "cer": round(jiwer.cer(reference, heard), 4), **v})
        print(f"{out_dir.name}  {file_name}  CER {rows[-1]['cer']:.3f}  heard: {heard}", flush=True)

    total_open = sum(vowels.values()) or 1
    summary = {
        "source": str(speaker.source if speaker else folder), "clips": len(rows),
        "cer": round(jiwer.cer(refs, hyps), 4), "wer": round(jiwer.wer(refs, hyps), 4),
        "open_vowels_kept": round(vowels["kept"] / total_open, 4),
        "open_vowels_flattened": round(vowels["flattened"] / total_open, 4),
        "open_vowel_count": sum(vowels.values()),
    }
    with open(out_dir / "clips.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({out_dir.name: summary}, indent=2, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    p.add_argument("--model", action="append", default=[],
                   help="[name=]path of a TTS model that says each text instead of the folder's "
                        "audio; give it several times to score several models")
    p.add_argument("--name", help="results folder name under outputs/eval, for a single run")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    items = read_items(args.folder)[:args.limit]
    listener = Listener()
    if not args.model:
        score(items, args.folder, ROOT / "outputs/eval" / (args.name or args.folder.name + "-real"), listener)
        return

    from speak import Speaker
    for spec in args.model:
        name, _, model = spec.rpartition("=")
        name = name or (args.name if len(args.model) == 1 else None) or Path(model).name
        speaker = Speaker(model, seed=args.seed)
        speaker.source = model
        score(items, args.folder, ROOT / "outputs/eval" / name, listener, speaker)
        del speaker
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
