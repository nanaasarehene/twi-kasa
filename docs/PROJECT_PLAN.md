# Plan

Deliverables: repo, shared folder (samples + prompts + checkpoint), report
(.docx, 4 pages max), live demo.

- [x] Repo and environment set up, `src/check_setup.py` passes
- [x] Choose model (MMS-aka, partly frozen fine-tune)
- [x] Choose data (WAXAL Twi TTS) and download it
- [x] Twi text clean-up, numbers checked against the speakers, tests pass
- [x] Forced alignment + cutting, tested on a 6-min recording (95% kept)
- [x] Baseline samples from the untouched model (`outputs/samples/base/`)
- [x] Training script with freezing patch, inference, scoring, Colab notebook
- [ ] Pick a speaker (`outputs/speakers/`)
- [ ] Push to GitHub, run the Colab notebook
- [ ] Score checkpoints, keep the best, generate samples
- [ ] Twi diagnostic sentences (ɛ/e, ɔ/o, tone pairs, vowel harmony) and listening test
- [ ] Report, slides, shared Drive folder

Things that could go wrong:
- Colab disconnects mid-training: everything is on Drive; re-run the cell, it resumes.
- Colab runs out of free GPU time: Kaggle notebooks as a backup.
- Fine-tuning makes the voice worse than the base model: every checkpoint is
  scored against the base model and the best one is kept.
- Not enough listeners: ask 5–8 Twi-speaking classmates to fill in a short
  Google Form.
