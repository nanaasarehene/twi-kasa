"""Turn the raw WAXAL Twi recordings into short, aligned training clips.

Most WAXAL recordings are far too long for VITS (median 16 s, some over 6 min).
For each recording we:
  1. clean the transcript and split it into sentences, and sentences into
     clauses at commas
  2. force-align every word to the audio with Meta's MMS aligner
  3. group neighbouring clauses into clips of at most `max_seconds`, cutting
     at sentence ends where possible and at commas only when needed
  4. keep each clip's alignment score so badly matched text can be filtered

Output (Hugging Face "audiofolder" layout, loads straight into `datasets`):
  data/processed/<split>/<clip>.wav       16 kHz mono
  data/processed/<split>/metadata.csv     file_name, text, model_text, speaker, ...

Usage:
  python src/prepare_data.py                     # everything
  python src/prepare_data.py --speakers 2 --limit 5
"""
import argparse
import csv
import io
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
import yaml

from twi_text import clean, for_alignment, for_model, split_clauses, split_sentences

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "validation", "test")
FIELDS = ["file_name", "text", "model_text", "speaker_id", "gender", "source_id",
          "start", "end", "duration", "score"]


class Aligner:
    def __init__(self, chunk_seconds):
        bundle = torchaudio.pipelines.MMS_FA
        self.sr = bundle.sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = bundle.get_model().eval().to(self.device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.chunk = int(chunk_seconds * self.sr)

    @torch.inference_mode()
    def __call__(self, wave, words):
        """Return [(start_s, end_s, score)] for each word in `words`."""
        # the aligner's attention is quadratic in length, so long recordings
        # are fed in chunks and the frame-level outputs stitched back together
        wave = wave.to(self.device)
        pieces = [self.model(wave[:, i:i + self.chunk])[0]
                  for i in range(0, wave.shape[1], self.chunk)]
        emission = torch.cat(pieces, dim=1).cpu()
        seconds_per_frame = wave.shape[1] / emission.shape[1] / self.sr
        spans = self.aligner(emission[0], self.tokenizer(words))
        return [(s[0].start * seconds_per_frame,
                 s[-1].end * seconds_per_frame,
                 sum(t.score * len(t) for t in s) / sum(len(t) for t in s))
                for s in spans]


def download(cfg, raw):
    """Fetch the three WAXAL Twi TTS parquet files if they aren't there yet."""
    from huggingface_hub import hf_hub_download
    for split in SPLITS:
        if not (raw / f"twi-{split}.parquet").exists():
            print(f"downloading {split} split ...", flush=True)
            path = hf_hub_download(cfg["dataset"]["hf_id"], repo_type="dataset",
                                   filename=f"data/TTS/twi/twi-{split}-00000.parquet")
            raw.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, raw / f"twi-{split}.parquet")


def load_audio(audio_bytes, target_sr):
    wave, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    wave = torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)
    return torchaudio.functional.resample(wave, sr, target_sr)


def phrase_spans(text, wave, aligner):
    """Align a transcript and return [(phrase, start_s, end_s, score, ends_sentence)].

    A phrase is a clause between commas/semicolons, so a long sentence can still be
    cut at a natural pause. Grouping later only cuts mid-sentence when it has to.
    """
    phrases = [(clause, k == len(clauses) - 1)
               for sentence in split_sentences(clean(text))
               for clauses in [split_clauses(sentence)]
               for k, clause in enumerate(clauses)]
    words, owner = [], []
    for i, (phrase, _) in enumerate(phrases):
        for w in phrase.split():
            a = for_alignment(w)
            if a:  # punctuation-only tokens have nothing to align
                words.append(a)
                owner.append(i)
    if not words:
        return []
    timed = aligner(wave, words)
    out = []
    for i, (phrase, ends_sentence) in enumerate(phrases):
        mine = [t for t, o in zip(timed, owner) if o == i]
        if mine:
            score = sum(t[2] for t in mine) / len(mine)
            out.append((phrase, mine[0][0], mine[-1][1], score, ends_sentence))
    return out


def group_into_clips(spans, total_s, cfg):
    """Join neighbouring phrases into clips no longer than max_seconds.

    Prefers to cut at the end of a sentence; cuts after a comma only when the
    sentence is too long to keep whole. Cut points sit in the pause between
    phrases: padded out from the speech, but never past the midpoint of the gap.
    """
    a = cfg["audio"]
    cuts = []
    for k, (_, start, end, _, _) in enumerate(spans):
        prev_end = spans[k - 1][2] if k else 0.0
        next_start = spans[k + 1][1] if k + 1 < len(spans) else total_s
        cuts.append((max(start - a["pad_before"], (prev_end + start) / 2),
                     min(end + a["pad_after"], (end + next_start) / 2)))

    clips, current = [], []
    for k in range(len(spans)):
        if current and cuts[k][1] - cuts[current[0]][0] > a["max_seconds"]:
            # back up to the last sentence end inside the clip, if there is one
            ends = [i for i, j in enumerate(current) if spans[j][4]]
            split = ends[-1] + 1 if ends else len(current)
            clips.append(current[:split])
            current = current[split:]
        current.append(k)
    if current:
        clips.append(current)

    for group in clips:
        start, end = cuts[group[0]][0], cuts[group[-1]][1]
        if not a["min_seconds"] <= end - start <= a["max_seconds"]:
            continue  # a single clause that is too long on its own, or a stray word
        score = sum(spans[k][3] for k in group) / len(group)
        if score < cfg["alignment"]["min_score"]:
            continue  # text and audio probably don't match (misread or skipped words)
        yield " ".join(spans[k][0] for k in group), start, end, score


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs/data.yaml")
    p.add_argument("--speakers", nargs="*", help="speaker ids to keep (default: all)")
    p.add_argument("--limit", type=int, help="only process this many recordings per split (for testing)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw = ROOT / cfg["paths"]["raw_dir"]
    out = ROOT / cfg["paths"]["processed_dir"]
    sr = cfg["audio"]["target_sample_rate"]
    download(cfg, raw)
    aligner = Aligner(cfg["alignment"]["chunk_seconds"])
    assert aligner.sr == sr, "aligner and model sample rates differ"

    for split in SPLITS:
        rows = pq.read_table(raw / f"twi-{split}.parquet").to_pylist()
        if args.speakers:
            rows = [r for r in rows if r["speaker_id"] in args.speakers]
        rows = rows[:args.limit] if args.limit else rows

        (out / split).mkdir(parents=True, exist_ok=True)
        kept_s = source_s = 0.0
        tic = time.time()
        with open(out / split / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for n, r in enumerate(rows, 1):
                wave = load_audio(r["audio"]["bytes"], sr)
                total_s = wave.shape[1] / sr
                source_s += total_s
                spans = phrase_spans(r["text"], wave, aligner)
                for j, (text, start, end, score) in enumerate(group_into_clips(spans, total_s, cfg)):
                    name = f"{r['id']}_{j:02d}.wav"
                    piece = wave[:, int(start * sr):int(end * sr)]
                    sf.write(out / split / name, piece.squeeze(0).numpy(), sr, subtype="PCM_16")
                    writer.writerow({
                        "file_name": name, "text": text, "model_text": for_model(text),
                        "speaker_id": r["speaker_id"], "gender": r["gender"], "source_id": r["id"],
                        "start": round(start, 3), "end": round(end, 3),
                        "duration": round(end - start, 3), "score": round(score, 4)})
                    kept_s += end - start
                if n % 25 == 0 or n == len(rows):
                    print(f"  {split}: {n}/{len(rows)} recordings, kept {kept_s/3600:.2f} h "
                          f"of {source_s/3600:.2f} h  ({time.time() - tic:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
