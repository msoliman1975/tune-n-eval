"""Converts raw sources to Markdown: PDFs with Docling, HTML with Trafilatura.

    bash run.sh parse [--force]
Writes parsed/{doc_id}.md and parsed/index.jsonl. A document whose text is too short is flagged
`low_text` (usually a scanned PDF with no text layer; M1 skips OCR) and left out of chunking.
"""

from __future__ import annotations

import argparse
import json

from . import ARTIFACTS
from .collect import RAW, load_sources

PARSED = ARTIFACTS / "parsed"
MIN_CHARS = 1500


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("؀" <= c <= "ۿ" for c in letters) / max(len(letters), 1)


def pdf_to_md(path) -> str:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    global _converter
    if "_converter" not in globals():
        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter.convert(path).document.export_to_markdown()


def html_to_md(path) -> str:
    import trafilatura

    html = path.read_bytes().decode("utf-8", errors="replace")
    return trafilatura.extract(html, output_format="markdown", include_tables=True, include_comments=False) or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-parse documents already parsed")
    a = ap.parse_args()

    PARSED.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in load_sources():
        raw = RAW / f"{src['doc_id']}.{src['format']}"
        out = PARSED / f"{src['doc_id']}.md"
        row = {"doc_id": src["doc_id"], "lang": src["lang"], "license": src.get("license", "unknown")}
        if not raw.exists():
            rows.append(row | {"status": "missing_raw"})
            continue
        if out.exists() and not a.force:
            text = out.read_text(encoding="utf-8")
        else:
            try:
                text = pdf_to_md(raw) if src["format"] == "pdf" else html_to_md(raw)
            except Exception as e:  # one bad file must not stop the batch
                rows.append(row | {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]})
                print(f"error   {src['doc_id']}  {rows[-1]['error']}", flush=True)
                continue
            out.write_text(text, encoding="utf-8")
        ratio = arabic_ratio(text)
        detected = "ar" if ratio > 0.5 else "en"
        status = "low_text" if len(text) < MIN_CHARS else "ok"
        rows.append(row | {"status": status, "chars": len(text), "arabic_ratio": round(ratio, 3),
                           "lang_mismatch": detected != src["lang"]})
        print(f"{status:7} {src['doc_id']}  {len(text):>8} chars  ar={ratio:.2f}", flush=True)
    with (PARSED / "index.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{sum(r['status'] == 'ok' for r in rows)}/{len(rows)} documents usable")


if __name__ == "__main__":
    main()
