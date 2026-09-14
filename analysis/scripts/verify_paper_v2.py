"""Check paper artifacts and render every PDF page for manual visual inspection.

Requires the optional PDF-review dependency: pip install pymupdf.
Run after generation and compilation: python -m analysis.scripts.verify_paper_v2
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper_v2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import pymupdf
    from PIL import Image, ImageDraw

    baseline = json.loads((RESULTS / "benchmark_baseline_seed42.json").read_text())
    current = json.loads((RESULTS / "benchmark_seed42.json").read_text())
    for key in baseline:
        if key not in {"speed_benchmark", "git_sha"}:
            assert baseline[key] == current[key], f"Changed financial result: {key}"
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert digest(ROOT / name) == expected, f"Changed source: {name}"
    for name, expected in manifest["figure_sha256"].items():
        assert digest(ROOT / "paper/figures/v2" / name) == expected, name
    assert digest(RESULTS / "environment.txt") == manifest["environment_sha256"]
    assert digest(RESULTS / "benchmark_seed42.json") == manifest["benchmark_sha256"]
    paths = pd.read_csv(RESULTS / "scenario_paths.csv")
    assert len(paths) == 1000 and paths.scenario_id.nunique() == 200
    cash = (
        paths.opening_cash
        + paths.operating_cash_generation
        - paths.lease_principal_cash_payment
        - paths.actual_mandatory_amortisation
        - paths.cash_sweep
        + paths.revolver_draw
        - paths.revolver_repayment
    )
    assert np.allclose(cash, paths.ending_cash, atol=1e-10)
    accor = pd.read_csv(RESULTS / "accor_results.csv")
    assert accor.icr_ifrs16.notna().sum() == 4
    assert accor.breach_ifrs16.sum() == 5 and accor.breach_frozen_gaap.sum() == 3
    tex = (ROOT / "paper/ifrs16_lbo_ssrn_v2.tex").read_text(encoding="utf-8")
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels)), "Duplicate LaTeX labels"
    figures = re.findall(r"\\includegraphics\[[^]]+\]\{([^}]+)\}", tex)
    assert len(figures) == len(set(figures)) == 6
    log = (ROOT / "paper/ifrs16_lbo_ssrn_v2.log").read_text(encoding="utf-8", errors="replace")
    prohibited = [
        r"undefined",
        r"Undefined control sequence",
        r"Missing character",
        r"Overfull",
        r"multiply defined",
        r"File .+ not found",
    ]
    for pattern in prohibited:
        assert not re.search(pattern, log, re.IGNORECASE), f"LaTeX diagnostic: {pattern}"
    pdf_path = ROOT / "paper/ifrs16_lbo_ssrn_v2.pdf"
    pdf = pymupdf.open(pdf_path)
    review = ROOT / "output/paper_v2_review"
    review.mkdir(parents=True, exist_ok=True)
    pages, thumbnails = [], []
    for index, page in enumerate(pdf):
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2), alpha=False)
        page_path = review / f"page_{index + 1:02d}.png"
        pixmap.save(str(page_path))
        text = page.get_text()
        assert len(text.strip()) > 20, f"Empty page {index + 1}"
        pages.append(
            {
                "page": index + 1,
                "text_characters": len(text),
                "vector_drawings": len(page.get_drawings()),
                "figure_captions": re.findall(r"Figure\s+[1-6]:", text),
                "render_sha256": digest(page_path),
            }
        )
        thumbnail = Image.open(page_path).convert("RGB")
        thumbnail.thumbnail((595, 842))
        thumbnails.append(thumbnail)
    full_text = "\n".join(page.get_text() for page in pdf)
    assert "??" not in full_text, "Unresolved PDF reference"
    for number in range(1, 7):
        assert re.search(rf"Figure\s+{number}:", full_text), f"Missing Figure {number} caption"
    # Vector figures must contain actual drawing content on their own PDFs.
    for figure in figures:
        figure_pdf = pymupdf.open(ROOT / "paper/figures/v2" / f"{figure}.pdf")
        assert len(figure_pdf) == 1 and len(figure_pdf[0].get_drawings()) > 10, figure
    for start in range(0, len(thumbnails), 4):
        sheet = Image.new("RGB", (1190, 1740), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for offset, thumbnail in enumerate(thumbnails[start : start + 4]):
            x, y = (offset % 2) * 595, (offset // 2) * 870
            sheet.paste(thumbnail, (x, y + 24))
            draw.text((x + 10, y + 5), f"Page {start + offset + 1}", fill="black")
        sheet.save(review / f"contact_{start // 4 + 1}.png")
    verification = {
        "source_commit": manifest["source_commit"],
        "source_files_dirty": manifest["source_files_dirty"],
        "baseline_financial_results_identical": True,
        "artifact_checksums_match": True,
        "pdf_sha256": digest(pdf_path),
        "latex_log_sha256": digest(ROOT / "paper/ifrs16_lbo_ssrn_v2.log"),
        "latex_errors_or_overfull_boxes": False,
        "figure_count": len(figures),
        "page_count": len(pdf),
        "pages": pages,
        "visual_review": "Every page rendered; manual inspection recorded in validation.json",
    }
    (RESULTS / "pdf_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(
        json.dumps(
            {"pages": len(pdf), "figures": len(figures), "review_directory": str(review)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
