"""Build the integrated Week 11 experiment report DOCX.

The source package on the desktop contains a UTF-8 Markdown report plus figure
PNGs.  This script converts that package into a polished Word document while
preserving the project's evidence-boundary language.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_ZIP = Path(r"<user_home>\Desktop\files.zip")
DEFAULT_PACKAGE_DIR = ROOT / "tmp" / "week11_report_package"
DEFAULT_OUTPUT = ROOT / "output" / "doc" / "Week11_Final_Experiment_Report.docx"


PRIMARY = RGBColor(31, 78, 121)
ACCENT = RGBColor(84, 130, 53)
MUTED = RGBColor(89, 89, 89)
FAIL = RGBColor(156, 0, 6)
PASS = RGBColor(0, 97, 0)
WARN = RGBColor(156, 101, 0)


def set_run_font(run, font_name: str = "Microsoft YaHei", size: Pt | None = None) -> None:
    run.font.name = font_name
    if size is not None:
        run.font.size = size
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.append(r_fonts)
    r_fonts.set(qn("w:eastAsia"), font_name)
    r_fonts.set(qn("w:ascii"), font_name)
    r_fonts.set(qn("w:hAnsi"), font_name)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text: str, bold: bool = False, color: RGBColor | None = None, size: Pt = Pt(8)) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run(text)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    set_run_font(run, size=size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.7)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(1.7)
    section.right_margin = Cm(1.7)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.12

    for name, size, color in [
        ("Title", 22, PRIMARY),
        ("Heading 1", 16, PRIMARY),
        ("Heading 2", 13, RGBColor(46, 96, 135)),
        ("Heading 3", 11.5, RGBColor(68, 68, 68)),
        ("Heading 4", 10.5, RGBColor(68, 68, 68)),
    ]:
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.paragraph_format.space_before = Pt(8 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(5)


def prepare_package(zip_path: Path, package_dir: Path) -> Path:
    md_path = package_dir / "Week11_Final_Experiment_Report.md"
    if md_path.exists():
        return md_path
    if not zip_path.exists():
        raise FileNotFoundError(f"Package zip not found: {zip_path}")
    package_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(package_dir)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown report not found after extraction: {md_path}")
    return md_path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def add_inline_runs(paragraph, text: str, default_bold: bool = False) -> None:
    token_re = re.compile(r"(`[^`]+`|\*\*.*?\*\*)")
    cursor = 0
    for match in token_re.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor : match.start()])
            run.bold = default_bold
            set_run_font(run)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            set_run_font(run)
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            r_pr = run._element.get_or_add_rPr()
            r_fonts = r_pr.rFonts
            if r_fonts is None:
                r_fonts = OxmlElement("w:rFonts")
                r_pr.append(r_fonts)
            r_fonts.set(qn("w:eastAsia"), "Consolas")
            r_fonts.set(qn("w:ascii"), "Consolas")
            r_fonts.set(qn("w:hAnsi"), "Consolas")
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        run.bold = default_bold
        set_run_font(run)


def split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip().replace("<br>", "\n") for cell in line.split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells)


def add_markdown_table(document: Document, table_lines: list[str]) -> None:
    if len(table_lines) < 2:
        return
    rows = [split_table_row(line) for line in table_lines if not is_table_separator(line)]
    if not rows:
        return
    max_cols = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=max_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for r_idx, row in enumerate(rows):
        for c_idx in range(max_cols):
            text = row[c_idx] if c_idx < len(row) else ""
            cell = table.cell(r_idx, c_idx)
            is_header = r_idx == 0
            set_cell_text(cell, text, bold=is_header, size=Pt(7.5 if max_cols > 4 else 8.5))
            if is_header:
                set_cell_shading(cell, "D9EAF7")
            elif r_idx % 2 == 0:
                set_cell_shading(cell, "F7F9FB")
    document.add_paragraph()


def resolve_image(package_dir: Path, image_ref: str) -> Path | None:
    candidates = [
        package_dir / image_ref,
        package_dir / Path(image_ref).name,
        package_dir / "figs" / Path(image_ref).name,
        ROOT / "output" / "week11" / "plots" / Path(image_ref).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def add_image(document: Document, package_dir: Path, alt: str, image_ref: str) -> None:
    image_path = resolve_image(package_dir, image_ref)
    if not image_path:
        paragraph = document.add_paragraph(style="Intense Quote")
        add_inline_runs(paragraph, f"[图片缺失: {alt} -> {image_ref}]")
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(6.4))
    caption = document.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption.add_run(alt)
    caption_run.italic = True
    caption_run.font.color.rgb = MUTED
    set_run_font(caption_run, size=Pt(9))


def add_callout(document: Document, title: str, body: str, fill: str = "EAF4EA") -> None:
    table = document.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(title)
    run.bold = True
    run.font.color.rgb = ACCENT
    set_run_font(run, size=Pt(10))
    paragraph.add_run("\n")
    add_inline_runs(paragraph, body)
    document.add_paragraph()


def add_cover(document: Document, title: str, zip_path: Path, output_path: Path) -> None:
    score_rows = read_csv_rows(ROOT / "output" / "week11" / "metrics" / "week11_score_estimate.csv")
    audit_rows = read_csv_rows(ROOT / "output" / "week11" / "metrics" / "week11_acceptance_audit.csv")
    pass_count = sum(1 for row in audit_rows if row.get("status") == "pass")
    fail_count = sum(1 for row in audit_rows if row.get("status") == "fail")
    score_band = score_rows[0].get("estimated_project_score_band", "not available") if score_rows else "not available"

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = p.add_run(title)
    title_run.bold = True
    title_run.font.color.rgb = PRIMARY
    set_run_font(title_run, size=Pt(22))

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle.add_run("整合版 DOCX 报告")
    subtitle_run.font.color.rgb = MUTED
    set_run_font(subtitle_run, size=Pt(12))

    document.add_paragraph()
    meta = [
        ("来源压缩包", str(zip_path)),
        ("项目目录", str(ROOT)),
        ("输出文件", str(output_path)),
        ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Week11 验收", f"{pass_count} pass / {fail_count} fail / {len(audit_rows)} total"),
        ("项目分数口径", score_band),
    ]
    table = document.add_table(rows=len(meta), cols=2)
    table.style = "Table Grid"
    for idx, (key, value) in enumerate(meta):
        set_cell_text(table.cell(idx, 0), key, bold=True, size=Pt(9))
        set_cell_text(table.cell(idx, 1), value, size=Pt(9))
        set_cell_shading(table.cell(idx, 0), "D9EAF7")

    document.add_paragraph()
    add_callout(
        document,
        "证据边界说明",
        "本文档按 Week10/Week11 的最终口径整合：保留负向发现、明确 post-hoc 与 prerun-frozen 边界，"
        "并把 paper_safe_claim / paper_forbidden_claim 作为论文写作约束。",
        fill="EAF4EA",
    )
    document.add_page_break()


def add_quality_appendix(document: Document) -> None:
    document.add_heading("附录 C: DOCX 整合一致性校验", level=2)

    patch_rows = read_csv_rows(ROOT / "output" / "week11" / "tables" / "table_week11_patch_summary.csv")
    audit_rows = read_csv_rows(ROOT / "output" / "week11" / "metrics" / "week11_acceptance_audit.csv")
    final_audit = ROOT / "output" / "final_artifact_audit.csv"

    add_callout(
        document,
        "最终口径",
        "本 DOCX 以 files.zip 中的 Week11_Final_Experiment_Report.md 为主稿，"
        "并用 output/week11 下的 acceptance audit、score estimate 和 patch summary 进行交叉核对。",
        fill="FFF2CC",
    )

    if patch_rows:
        document.add_heading("C.1 Week11 patch summary", level=3)
        table = document.add_table(rows=len(patch_rows) + 1, cols=len(patch_rows[0]))
        table.style = "Table Grid"
        headers = list(patch_rows[0].keys())
        for c_idx, header in enumerate(headers):
            set_cell_text(table.cell(0, c_idx), header, bold=True, size=Pt(7.5))
            set_cell_shading(table.cell(0, c_idx), "D9EAF7")
        for r_idx, row in enumerate(patch_rows, start=1):
            for c_idx, header in enumerate(headers):
                set_cell_text(table.cell(r_idx, c_idx), row.get(header, ""), size=Pt(7.5))
        document.add_paragraph()

    if audit_rows:
        document.add_heading("C.2 Week11 acceptance audit", level=3)
        headers = ["patch", "hypothesis", "metric", "observed_value", "threshold", "status"]
        table = document.add_table(rows=len(audit_rows) + 1, cols=len(headers))
        table.style = "Table Grid"
        for c_idx, header in enumerate(headers):
            set_cell_text(table.cell(0, c_idx), header, bold=True, size=Pt(7))
            set_cell_shading(table.cell(0, c_idx), "D9EAF7")
        for r_idx, row in enumerate(audit_rows, start=1):
            for c_idx, header in enumerate(headers):
                color = None
                if header == "status":
                    color = PASS if row.get(header) == "pass" else FAIL
                set_cell_text(table.cell(r_idx, c_idx), row.get(header, ""), color=color, size=Pt(6.7))
            if r_idx % 2 == 0:
                for c_idx in range(len(headers)):
                    set_cell_shading(table.cell(r_idx, c_idx), "F7F9FB")
        document.add_paragraph()

    if final_audit.exists():
        document.add_heading("C.3 Artifact audit reference", level=3)
        paragraph = document.add_paragraph()
        add_inline_runs(paragraph, f"Final artifact audit source: `{final_audit}`.")


def convert_markdown(document: Document, md_path: Path, package_dir: Path) -> str:
    lines = md_path.read_text(encoding="utf-8-sig").splitlines()
    title = "Week11 Final Experiment Report"
    idx = 0
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            add_markdown_table(document, table_lines)
            table_lines = []

    while idx < len(lines):
        raw = lines[idx]
        line = raw.rstrip()

        if line.strip().startswith("```"):
            flush_table()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                if code_lines:
                    for code_line in code_lines:
                        p = document.add_paragraph()
                        run = p.add_run(code_line)
                        set_run_font(run, "Consolas", Pt(8.5))
                    document.add_paragraph()
            idx += 1
            continue

        if in_code:
            code_lines.append(line)
            idx += 1
            continue

        if not line.strip():
            flush_table()
            idx += 1
            continue

        if line.lstrip().startswith("|"):
            table_lines.append(line)
            idx += 1
            continue

        flush_table()

        if re.fullmatch(r"\s*-{3,}\s*", line):
            document.add_paragraph()
            idx += 1
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            add_image(document, package_dir, image_match.group(1), image_match.group(2))
            idx += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            if level == 1 and title == "Week11 Final Experiment Report":
                title = text
                idx += 1
                continue
            if level <= 4:
                document.add_heading(text, level=level)
            else:
                p = document.add_paragraph()
                run = p.add_run(text)
                run.bold = True
                run.font.color.rgb = MUTED
                set_run_font(run, size=Pt(10))
            idx += 1
            continue

        if line.startswith(">"):
            quote = line.lstrip(">").strip()
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.45)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(quote)
            run.italic = True
            run.font.color.rgb = MUTED
            set_run_font(run, size=Pt(9.5))
            idx += 1
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet_match:
            p = document.add_paragraph(style="List Bullet")
            add_inline_runs(p, bullet_match.group(1))
            idx += 1
            continue

        number_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if number_match:
            p = document.add_paragraph(style="List Number")
            add_inline_runs(p, number_match.group(1))
            idx += 1
            continue

        p = document.add_paragraph()
        add_inline_runs(p, line)
        idx += 1

    flush_table()
    return title


def build_docx(zip_path: Path, package_dir: Path, output_path: Path) -> None:
    md_path = prepare_package(zip_path, package_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    configure_document(document)
    title = "Week11 Final Experiment Report"
    for line in md_path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^#\s+(.*)$", line.strip())
        if match:
            title = match.group(1).strip()
            break

    add_cover(document, title, zip_path, output_path)
    title = convert_markdown(document, md_path, package_dir)
    add_quality_appendix(document)

    # A final narrow section keeps wide appendix tables from inheriting odd
    # page settings if Word decides to insert section breaks during editing.
    document.add_section(WD_SECTION.CONTINUOUS)

    document.core_properties.title = title
    document.core_properties.subject = "Week11 hetero-transfer experiment integrated report"
    document.core_properties.author = "Codex / Raphael"
    document.core_properties.comments = "Generated from files.zip and Week11 project artifacts."
    document.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_PACKAGE_ZIP)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_docx(args.zip, args.package_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
