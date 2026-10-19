from twi_text import MODEL_EXTRA, MODEL_LETTERS, clean, for_alignment, for_model, number_to_words


def test_waxal_brackets_removed():
    assert clean("[10] Du [×] ahodoɔ [10] du [=] ma yɛn [100] ɔha.") == "Du ahodoɔ du ma yɛn ɔha."


def test_open_vowels_survive_and_lowercase():
    assert for_model("Ɛyɛ Ɔkɔɔ") == "ɛyɛ ɔkɔɔ"


def test_output_only_uses_model_alphabet():
    text = "Ɔpanin Kwamena’s “Mempɛasɛm” – Jesus Zion, GH₵ 25; 2026 (á, ã)!"
    assert set(for_model(text)) <= MODEL_LETTERS | MODEL_EXTRA


def test_numbers_match_waxal_speakers():
    # forms taken from how the WAXAL speakers read the bracketed numbers
    assert number_to_words(3) == "mmeɛnsa"
    assert number_to_words(14) == "du-nnan"
    assert number_to_words(25) == "aduonu-nnum"
    assert number_to_words(100) == "ɔha"
    assert number_to_words(107) == "ɔha ne nson"
    assert number_to_words(450) == "ahanan ne aduonum"
    assert number_to_words(1000) == "apem"
    assert number_to_words(1960) == "apem ahankron ne aduosia"
    assert number_to_words(2006) == "mpenu ne nsia"
    assert number_to_words(8056) == "mpem nwɔtwe ne aduonum-nsia"


def test_numbers_inside_text():
    assert for_model("Mewɔ GH₵ 25.") == "mewɔ sidi aduonu-nnum"


def test_alignment_text_is_plain_ascii():
    assert for_alignment("Wɔ,") == "wo"
    assert for_alignment("ɛyɛ") == "eye"
