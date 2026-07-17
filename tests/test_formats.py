import json
from pathlib import Path

import pytest

from engine.formats import (
    export_to_docx,
    export_to_json,
    export_to_txt,
    load_source_file,
    parse_json,
)

TIBETAN = "སངས་རྒྱས་ཆོས་དང་ཚོགས་ཀྱི་མཆོག་རྣམས་ལ།"


# ── load_source_file ──────────────────────────────────────────────────────

def test_load_txt(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text(f"{TIBETAN}\n\nSecond paragraph.", encoding="utf-8")
    segments = load_source_file(str(path))
    assert segments == [
        {"source": TIBETAN, "target": ""},
        {"source": "Second paragraph.", "target": ""},
    ]


def test_load_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "in.docx"
    doc = docx.Document()
    doc.add_paragraph(TIBETAN)
    doc.add_paragraph("Second paragraph.")
    doc.add_paragraph("")  # blank paragraphs should be dropped, not turned into segments
    doc.save(str(path))

    segments = load_source_file(str(path))
    assert segments == [
        {"source": TIBETAN, "target": ""},
        {"source": "Second paragraph.", "target": ""},
    ]


def test_load_pdf(tmp_path):
    pytest.importorskip("pypdf")
    reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.lib.pagesizes import letter

    path = tmp_path / "in.pdf"
    c = reportlab_canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 700, "Hello world.")
    c.save()

    segments = load_source_file(str(path))
    assert len(segments) == 1
    assert "Hello world." in segments[0]["source"]
    assert segments[0]["target"] == ""


def test_load_unsupported_extension_raises(tmp_path):
    path = tmp_path / "in.rtf"
    path.write_text("whatever", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_source_file(str(path))


def test_load_empty_file_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError, match="No text could be extracted"):
        load_source_file(str(path))


def test_load_respects_max_chars(tmp_path):
    path = tmp_path / "in.txt"
    long_text = ("a" * 50 + "। ") * 10
    path.write_text(long_text, encoding="utf-8")
    segments = load_source_file(str(path), max_chars=100)
    assert len(segments) > 1
    assert all(len(s["source"]) <= 110 for s in segments)


# ── the committed sample files (samples/sample_tibetan.*) ─────────────────

SAMPLES_DIR = Path(__file__).parent.parent / "samples"


@pytest.mark.parametrize("ext", ["txt", "docx", "pdf"])
def test_sample_files_load_successfully(ext):
    path = SAMPLES_DIR / f"sample_tibetan.{ext}"
    segments = load_source_file(str(path))
    assert len(segments) > 0
    assert all(seg["target"] == "" for seg in segments)
    combined = "".join(seg["source"] for seg in segments)
    assert "སངས་རྒྱས" in combined  # "Buddha" appears in the refuge verse


# ── parse_json ─────────────────────────────────────────────────────────────

def test_parse_json_valid_list():
    content = json.dumps([{"source": "src1", "target": "tgt1"}, {"source": "src2"}])
    segments = parse_json(content)
    assert segments == [
        {"source": "src1", "target": "tgt1"},
        {"source": "src2", "target": ""},
    ]


def test_parse_json_accepts_bytes():
    content = json.dumps([{"source": "src1", "target": "tgt1"}]).encode("utf-8")
    segments = parse_json(content)
    assert segments == [{"source": "src1", "target": "tgt1"}]


def test_parse_json_rejects_non_list():
    with pytest.raises(ValueError, match="non-empty list"):
        parse_json(json.dumps({"source": "a"}))


def test_parse_json_rejects_empty_list():
    with pytest.raises(ValueError, match="non-empty list"):
        parse_json(json.dumps([]))


def test_parse_json_rejects_missing_source():
    with pytest.raises(ValueError, match="Item 0 is missing 'source'"):
        parse_json(json.dumps([{"target": "only a target"}]))


def test_parse_json_rejects_non_dict_item():
    with pytest.raises(ValueError, match="Item 1 is missing 'source'"):
        parse_json(json.dumps([{"source": "ok"}, "not a dict"]))


# ── export_to_txt / export_to_docx / export_to_json ───────────────────────

def test_export_to_txt_joins_nonempty_targets():
    segments = [{"source": "s1", "target": "t1"}, {"source": "s2", "target": ""}, {"source": "s3", "target": "t3"}]
    assert export_to_txt(segments, "target") == "t1\n\nt3\n"


def test_export_to_txt_all_empty_returns_empty_string():
    segments = [{"source": "s1", "target": ""}, {"source": "s2", "target": "  "}]
    assert export_to_txt(segments, "target") == ""


def test_export_to_txt_source_field():
    segments = [{"source": "s1", "target": "t1"}, {"source": "s2", "target": "t2"}]
    assert export_to_txt(segments, "source") == "s1\n\ns2\n"


def test_export_to_docx_roundtrips_through_load(tmp_path):
    docx = pytest.importorskip("docx")
    segments = [{"source": "s1", "target": "First translation."}, {"source": "s2", "target": "Second translation."}]
    data = export_to_docx(segments, "target")
    assert isinstance(data, bytes) and len(data) > 0

    out_path = tmp_path / "out.docx"
    out_path.write_bytes(data)
    doc = docx.Document(str(out_path))
    texts = [p.text for p in doc.paragraphs if p.text.strip()]
    assert texts == ["First translation.", "Second translation."]


def test_export_to_docx_skips_empty_targets():
    segments = [{"source": "s1", "target": ""}, {"source": "s2", "target": "kept"}]
    data = export_to_docx(segments, "target")
    docx = pytest.importorskip("docx")
    import io
    doc = docx.Document(io.BytesIO(data))
    texts = [p.text for p in doc.paragraphs if p.text.strip()]
    assert texts == ["kept"]


def test_export_to_json_roundtrip():
    segments = [{"source": "s1", "target": "t1"}, {"source": "s2", "target": ""}]
    text = export_to_json(segments)
    assert json.loads(text) == segments


def test_export_to_json_preserves_unicode_without_escaping():
    segments = [{"source": TIBETAN, "target": ""}]
    text = export_to_json(segments)
    assert TIBETAN in text  # ensure_ascii=False
