# kasa – a Twi text-to-speech system

*Kasa* is the Twi word for speech. This repo has everything I used to build a
speech synthesiser that reads Twi text aloud: data preparation, training,
inference and evaluation.

The variety is the Asante dialect of Twi. Wherever this repo says "Twi",
it means Asante Twi.

## How it works

1. **Data.** The WAXAL Twi TTS recordings (University of Ghana / Google
   Research, CC BY 4.0) come as long readings. `prepare_data.py` lines up
   every word with the audio using a forced aligner, then cuts the recordings
   into sentence clips of 15 s or less.
2. **Text.** `twi_text.py` cleans the text. It spells numbers out the way the
   WAXAL speakers say them and keeps ɛ and ɔ. Everything is reduced to the
   letters the model knows.
3. **Model.** Meta's MMS-TTS Akan model (VITS) is fine-tuned on one speaker,
   with its text encoder and duration predictor frozen, so it keeps what it
   knows about Twi pronunciation while learning the new voice.
4. **Scoring.** An Akan speech recogniser transcribes the output. We measure
   character error rate and how often ɛ/ɔ survive, and compare against both
   the base model and real recordings.

The reasoning behind each choice is in `docs/DECISIONS.md`.

## Layout

- `src/`: all the code, one file per job
  - `check_setup.py`: checks that every library is installed
  - `twi_text.py`: cleans Twi text (numbers to words, ɛ/ɔ, punctuation)
  - `prepare_data.py`: downloads, aligns, cuts and splits the recordings
  - `train.py`: fine-tunes the model and exports each checkpoint
  - `speak.py`: turns Twi text into a .wav file
  - `score.py`: measures how understandable the output is
- `configs/`: data and training settings
- `notebooks/train_colab.ipynb`: runs the full pipeline on a Colab GPU
- `tests/`: tests for the text cleaner
- `data/`, `runs/`, `checkpoints/`, `outputs/`: generated files, kept out of git
- `docs/`: design decisions, project plan, report drafts

## Setting up (local)

I developed this on an Intel MacBook with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch==2.2.2" "torchaudio==2.2.2" "numpy<2"
pip install -r requirements.txt
python src/check_setup.py
python -m pytest
```

The torch line is for Intel Macs, which stopped getting new PyTorch builds
after 2.2.2. On any other machine, install torch the normal way from
pytorch.org first. `requirements.lock.txt` lists the exact versions I had
installed when everything worked.

## Training (Colab)

Open `notebooks/train_colab.ipynb` in Colab with a T4 GPU and run it top to
bottom. Outputs are saved to `MyDrive/kasa`, so a disconnect doesn't lose
the run.

## Speaking

```bash
python src/speak.py "Akwaaba, wo ho te sɛn?" --out hello.wav
python src/speak.py --file data/external/unseen_sentences.txt --out-dir outputs/samples/kasa
```

`--model facebook/mms-tts-aka` uses the untouched base model, for comparison.

## Credits

| What | Source | Licence | Used for |
|---|---|---|---|
| WAXAL Twi TTS | University of Ghana, Google Research (`google/WaxalNLP`) | CC BY 4.0 | training and test audio |
| MMS-TTS Akan | Meta AI (`facebook/mms-tts-aka`, discriminator from `facebook/mms-tts`) | CC BY-NC 4.0 | base model |
| MMS forced aligner | Meta AI, via torchaudio `MMS_FA` | CC BY-NC 4.0 | cutting recordings into sentences |
| MMS-1b-all + Akan adapter | Meta AI (`facebook/mms-1b-all`) | CC BY-NC 4.0 | evaluation only |
| finetune-hf-vits | Y. Lacombe | MIT | training loop (with my freezing patch) |
| VITS | Kim et al., ICML 2021 | – | architecture |

Because the base model is CC BY-NC, the fine-tuned model is for
non-commercial use only. No external TTS service is used anywhere.
