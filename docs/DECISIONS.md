# Decisions

Notes to self on what I chose and why. Most of this ends up in the report.

## Variety: Twi (Asante)
It's my own dialect, so I can judge pronunciation and tone errors by ear
instead of guessing.

## Data: WAXAL Twi TTS
`google/WaxalNLP`, config `twi_tts`. Recorded by the University of Ghana for
Google Research's WAXAL project. CC BY 4.0. 10.93 h, 1,093 recordings,
4 speakers (2 female, 2 male, 2.5–3.1 h each), 48 kHz, with WAXAL's own
train / validation / test split.

Why this one:
- **Everyday Twi.** Days of the week, food, numbers, short stories. The base
  model (MMS) was trained on Bible recordings, so more Bible audio would teach
  it little. This teaches it new words and a conversational rhythm.
- **Spelling is intact.** 17,933 ɛ and 15,112 ɔ, with Asante spelling
  (awarefoɔ, deɛ).
- **Most permissive licence** of the options (CC BY 4.0).
- **A real test split** of the same speaker, so synthetic speech can be
  compared with real speech of the same sentences.

Problems and how I handled them:
- **Recordings are long.** Median 16 s, some over 6 minutes, and only ~1 h is
  under 15 s. VITS can't train on that. `prepare_data.py` force-aligns every
  word with Meta's MMS aligner (torchaudio `MMS_FA`), cuts in the pauses
  between sentences, and regroups sentences into clips of 15 s or less.
  Each clip keeps its alignment score so badly matched text can be dropped.
- **The aligner has no ɛ/ɔ.** For alignment only, ɛ→e and ɔ→o. The training
  text keeps ɛ/ɔ.
- **Bracketed notes in transcripts.** WAXAL writes the printed form in
  brackets before the spoken words: `[10] du [×] ahodoɔ`. The brackets are
  removed, keeping what was actually said.
- **Four voices.** MMS-aka is a single-voice model, so I train on one speaker.

Other options I looked at:
- BibleTTS Twi (9.6 h, one speaker, CC BY-SA). The published transcripts
  have every ɛ and ɔ deleted (0 of either in 3,247 clips). A repaired text
  exists (`danieldzikunuofmarvel/bibletts-asante-twi-repaired`), but it is
  still Bible language, which the base model has already seen.
- GhanaNLP `ghana-speech`: also Bible, non-commercial licence, several narrators.
- UGSpeechData: recorded for ASR (spontaneous, noisy), not TTS quality.
- Common Voice Twi: too little audio.

## Text: numbers follow the speakers
The model's alphabet has no digits, so numbers are spelled out. I first used
textbook forms, then checked them against every bracketed number in WAXAL
where the spoken form follows the digits (255 distinct number/reading pairs). The speakers say *mmeɛnsa* (3),
*nnan* (4), *nnum* (5), *aduoson* (70), *mpenu ne …* (2000s), and hyphenate
compounds (*aduonu-nnum*). `twi_text.py` now follows them, so numbers come
out the way the voice was trained to say them.

## Model: MMS-TTS-aka (VITS), partly frozen fine-tune
I start from `facebook/mms-tts-aka`, the only openly available pretrained
Akan TTS model. VITS is end-to-end: its HiFi-GAN-style decoder is the vocoder,
trained together with the rest.

The catch: one public attempt in the cohort found that a full fine-tune made
intelligibility *worse* than the untouched base model (CER 0.088 → 0.232).
My guess is that the text encoder and duration predictor, which carry most of
the pronunciation and timing knowledge, get disturbed when every weight is
updated on a small corpus.

So I freeze the text encoder and duration predictor and only adapt the parts
that shape the voice and audio (posterior encoder, flow, decoder), at a low
learning rate (2e-5). Every 500 steps the checkpoint is exported and scored
against the base model, and the best one is kept, even if it's early.

The freezing is a small patch to finetune-hf-vits (`src/train.py`). The
harness itself has no option for it.

Rejected: SpeechT5 + HiFi-GAN (starts from English, likely less natural) and
Coqui VITS from an English checkpoint (needs far more GPU time).

## Evaluation
- **CER/WER** from an Akan speech recogniser (`facebook/mms-1b-all` + `aka`
  adapter), on held-out test sentences, for: real recordings (the ceiling),
  the base model, and each checkpoint.
- **ɛ/ɔ kept**: share of ɛ/ɔ in the text that the recogniser hears as ɛ/ɔ
  rather than e/o.
- Listening test with Twi speakers for naturalness, and my own notes on
  tone and vowel harmony errors.

## Compute
Code and quick checks on an Intel Mac (CPU, slow internet). Data
preparation, training and scoring on a Colab T4, with outputs on Google Drive.
