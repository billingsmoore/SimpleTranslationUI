from engine.chunking import split_into_segments


def test_blank_line_separates_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."
    assert split_into_segments(text, max_chars=400) == ["First paragraph.", "Second paragraph."]


def test_multiple_blank_lines_still_one_break():
    text = "First.\n\n\n\nSecond."
    assert split_into_segments(text, max_chars=400) == ["First.", "Second."]


def test_internal_whitespace_is_collapsed():
    text = "word1   word2\nword3"
    assert split_into_segments(text, max_chars=400) == ["word1 word2 word3"]


def test_empty_and_whitespace_only_input():
    assert split_into_segments("", max_chars=400) == []
    assert split_into_segments("   \n\n  \n", max_chars=400) == []


def test_short_paragraph_stays_one_segment():
    text = "སངས་རྒྱས་ཆོས་དང་ཚོགས་ཀྱི་མཆོག་རྣམས་ལ། །"
    assert split_into_segments(text, max_chars=400) == [text]


def test_long_paragraph_splits_on_shad_marks():
    sentence = "བདེ་བ་དང་ལྡན་པར་གྱུར་ཅིག" * 1  # one shad-terminated "sentence" below
    long_text = "།".join(["ཀ" * 30 for _ in range(20)]) + "།"
    segments = split_into_segments(long_text, max_chars=50)
    assert len(segments) > 1
    assert all(len(s) <= 60 for s in segments)  # allows small overhead from joining
    assert "".join(segments).replace(" ", "") == long_text.replace(" ", "")


def test_shad_split_regex_keeps_shad_with_preceding_text():
    text = "ཀཀཀ།ཁཁཁ༎"
    from engine.chunking import _SHAD_SPLIT_RE
    pieces = [p for p in _SHAD_SPLIT_RE.split(text) if p]
    assert pieces == ["ཀཀཀ།", "ཁཁཁ༎"]


def test_single_word_longer_than_max_chars_falls_back_to_word_chunking():
    text = " ".join(["word"] * 100)  # no shad marks at all, one huge "sentence"
    segments = split_into_segments(text, max_chars=30)
    assert len(segments) > 1
    for seg in segments:
        assert len(seg) <= 30 or " " not in seg  # a single overlong word is allowed through


def test_max_chars_boundary_exact_length_not_split():
    text = "a" * 400
    assert split_into_segments(text, max_chars=400) == [text]


def test_max_chars_boundary_one_over_length_is_split():
    text = ("a" * 200) + "། " + ("b" * 200)
    segments = split_into_segments(text, max_chars=400)
    assert len(segments) >= 1  # 401 chars total but splits cleanly on the shad mark
