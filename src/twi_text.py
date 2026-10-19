"""Text clean-up for Twi.

Three jobs:
  clean()          tidy a raw WAXAL transcript but keep its punctuation and case
  for_model()      turn any Twi text into what the MMS-aka model can read
  for_alignment()  turn a word into what the MMS forced aligner can read

The same for_model() runs at training time and at inference time, so the model
never sees text in a form it wasn't trained on.
"""
import re
import unicodedata

# Every letter the MMS-aka tokenizer knows (it also has 'á', '2', '3', which we never emit).
MODEL_LETTERS = set("abdefghiklmnoprstuwyɛɔ")
MODEL_EXTRA = set("'- ")

# Letters that aren't part of Twi spelling but turn up in names and loanwords.
# Nearest Twi sounds, so the model at least says *something* close.
LOAN_LETTERS = {"c": "k", "j": "gy", "q": "k", "v": "f", "x": "ks", "z": "s"}

APOSTROPHES = "’‘ʼ`´"


def clean(text):
    """Tidy a raw transcript. Keeps case and punctuation so sentences can still be split."""
    text = unicodedata.normalize("NFC", text)
    # WAXAL writes the printed form of numbers and symbols in brackets and then
    # the spoken words: "[10] du [×] ahodoɔ" -> "du ahodoɔ". Keep only what was said.
    text = re.sub(r"\[[^\]]*\]", " ", text)
    for ch in APOSTROPHES:
        text = text.replace(ch, "'")
    text = re.sub(r"[“”\"]", "", text)
    text = text.replace("​", "").replace("\xa0", " ").replace("–", " - ")
    text = re.sub(r"\s+", " ", text).strip()
    # brackets removal can leave " ." or " ,"
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def split_sentences(text):
    """Split on sentence-final punctuation. Abbreviations are rare enough in WAXAL to ignore."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_clauses(sentence):
    """Split a sentence at commas, semicolons and colons, where speakers usually pause."""
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    return [p.strip() for p in parts if p.strip()]


# --- numbers ---------------------------------------------------------------
# Spellings follow how the WAXAL speakers actually read numbers out (the
# corpus writes "[25] aduonu-nnum", "[1965] apem aha-nkron ne aduosia nnum"),
# not textbook forms, so the model hears numbers the way it was trained.

UNITS = ["", "baako", "mmienu", "mmeɛnsa", "nnan", "nnum", "nsia", "nson", "nwɔtwe", "nkron"]
TENS = ["", "du", "aduonu", "aduasa", "aduanan", "aduonum", "aduosia", "aduoson", "aduowɔtwe", "aduokron"]
HUNDREDS = ["", "ɔha", "ahaanu", "ahaasa", "ahanan", "ahanum", "ahansia", "ahanson", "ahanwɔtwe", "ahankron"]


def _below_hundred(n):
    tens, units = divmod(n, 10)
    if not tens:
        return UNITS[units]
    return TENS[tens] + ("-" + UNITS[units] if units else "")


def _below_thousand(n):
    hundreds, rest = divmod(n, 100)
    if not hundreds:
        return _below_hundred(rest)
    return HUNDREDS[hundreds] + (" ne " + _below_hundred(rest) if rest else "")


def number_to_words(n):
    """Cardinal number in Twi words, 0 to 999,999,999."""
    if n == 0:
        return "hwee"
    millions, rest = divmod(n, 1_000_000)
    thousands, rest = divmod(rest, 1000)
    words = []
    if millions:
        words.append("ɔpepem" if millions == 1 else "mpepem " + _below_thousand(millions))
    if thousands:
        words.append({1: "apem", 2: "mpenu"}.get(thousands) or "mpem " + _below_thousand(thousands))
    if rest:
        # speakers say "ne" before a small remainder but not before hundreds:
        # 2006 "mpenu ne nsia", 1900 "apem ahankron"
        if words and rest < 100:
            words.append("ne")
        words.append(_below_thousand(rest))
    return " ".join(words)


def _spell_numbers(text):
    text = re.sub(r"(?:GH)?₵\s*(\d+)", r"sidi \1", text)  # GH₵ 25 -> sidi 25
    text = re.sub(r"(\d+)\s*%", lambda m: m.group(1) + " ɔha mu", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,000 -> 1000
    return re.sub(r"\d+", lambda m: f" {number_to_words(int(m.group()))} ", text)


# --- model and aligner input ------------------------------------------------

def for_model(text):
    """Any Twi text -> lowercase string using only the model's alphabet."""
    text = clean(text)
    text = _spell_numbers(text)
    text = text.lower()  # Python maps Ɛ->ɛ and Ɔ->ɔ correctly
    # strip tone marks and nasal tildes (á, ã, õ): everyday spelling doesn't write them
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFC", text)
    text = "".join(LOAN_LETTERS.get(ch, ch) for ch in text)
    # punctuation becomes a word break; the model has no symbol for pauses
    text = "".join(ch if ch in MODEL_LETTERS or ch in MODEL_EXTRA else " " for ch in text)
    text = re.sub(r"\s-|-\s", " ", text)  # a spaced dash is a pause, not a compound
    return re.sub(r"\s+", " ", text).strip(" -'")


def for_alignment(word):
    """One word -> plain a-z for the MMS aligner, which has no ɛ or ɔ."""
    word = for_model(word).replace("ɛ", "e").replace("ɔ", "o")
    return re.sub(r"[^a-z']", "", word)
