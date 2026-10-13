"""Renders a Markdown document to PDF for client delivery.

Lives in the repo rather than a scratchpad on purpose: every client note in
``client_correspondence/`` ships as an .md/.pdf pair, and the converter that
made the earlier PDFs was a throwaway that no longer exists. That is not
reproducible for the client, and it means a doc edited later cannot be
re-rendered.

Supports the subset of Markdown these notes actually use: headings, bold,
italic, inline code, fenced code blocks, bullet and numbered lists, tables,
blockquotes, and horizontal rules.

Usage::

    python scripts/md_to_pdf.py client_correspondence/Some_Note.md
    python scripts/md_to_pdf.py input.md --out somewhere/output.pdf
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ReportLab's built-in fonts use WinAnsiEncoding. Characters outside it render
# as black boxes -- which is how a note goes out looking broken. Map the ones
# these documents actually contain to safe equivalents rather than hoping.
CHAR_FIXES: tuple[tuple[str, str], ...] = (
    ("→", "-&gt;"), ("←", "&lt;-"), ("⇒", "=&gt;"),
    ("≤", "&lt;="), ("≥", "&gt;="), ("≠", "!="),
    ("≈", "~"), ("×", "x"), ("−", "-"),
    ("✓", "yes"), ("✗", "no"), ("•", "-"),
    ("²", "<super>2</super>"), ("³", "<super>3</super>"),
    ("η", "eta"), ("α", "alpha"), ("χ", "chi"),
    ("‑", "-"), ("…", "..."),
)


def _inline(text: str) -> str:
    """Markdown inline formatting -> ReportLab's mini-HTML."""
    out = html.escape(text)
    # Code first: its contents must not be re-interpreted as emphasis.
    out = re.sub(r"`([^`]+)`",
                 r'<font face="Courier" size="9" color="#b03030">\1</font>', out)
    out = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                 r'<link href="\2" color="#1a4f8a">\1</link>', out)
    for bad, good in CHAR_FIXES:
        out = out.replace(bad, good)
    return out


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
        leading=14, spaceAfter=7, alignment=TA_LEFT,
    )
    return {
        "body": body,
        "h1": ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold",
                             fontSize=17, leading=21, spaceBefore=4, spaceAfter=10,
                             textColor=colors.HexColor("#12243a")),
        "h2": ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold",
                             fontSize=13, leading=17, spaceBefore=15, spaceAfter=7,
                             textColor=colors.HexColor("#1a3550")),
        "h3": ParagraphStyle("H3", parent=body, fontName="Helvetica-Bold",
                             fontSize=11, leading=15, spaceBefore=11, spaceAfter=5),
        "code": ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=8.5,
                               leading=11, leftIndent=10, spaceAfter=3, spaceBefore=3,
                               textColor=colors.HexColor("#233")),
        "quote": ParagraphStyle("Quote", parent=body, fontName="Helvetica-Oblique",
                                leftIndent=14, rightIndent=10, spaceBefore=6,
                                spaceAfter=8, textColor=colors.HexColor("#333")),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8.5, leading=11,
                               spaceAfter=0),
        "cellhdr": ParagraphStyle("CellH", parent=body, fontSize=8.5, leading=11,
                                  spaceAfter=0, fontName="Helvetica-Bold",
                                  textColor=colors.white),
    }


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _make_table(rows: list[list[str]], st: dict[str, ParagraphStyle], width: float) -> Table:
    header, *body_rows = rows
    ncols = max(len(header), max((len(r) for r in body_rows), default=0))
    data = []
    for i, row in enumerate(rows):
        padded = row + [""] * (ncols - len(row))
        style = st["cellhdr"] if i == 0 else st["cell"]
        data.append([Paragraph(_inline(c), style) for c in padded])

    tbl = Table(data, colWidths=[width / ncols] * ncols, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3550")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f5f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c2cbd4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def build(md: str, out_path: Path, title: str) -> None:
    st = _styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=title, author="Aegora Isle",
    )
    avail = doc.width
    story: list[object] = []
    lines = md.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            for cl in block:
                safe = html.escape(cl).replace(" ", "&nbsp;") or "&nbsp;"
                story.append(Paragraph(safe, st["code"]))
            story.append(Spacer(1, 6))
            continue

        # Table: a header row followed by a --- separator
        if stripped.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = [_split_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            story.append(_make_table(rows, st, avail))
            story.append(Spacer(1, 9))
            continue

        # Horizontal rule
        if re.match(r"^(\*\*\*|---|___)\s*$", stripped):
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=colors.HexColor("#c2cbd4")))
            story.append(Spacer(1, 7))
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            key = "h1" if level == 1 else "h2" if level == 2 else "h3"
            if level == 1 and any(isinstance(s, Paragraph) for s in story):
                story.append(PageBreak())
            story.append(Paragraph(_inline(m.group(2)), st[key]))
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            quoted = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quoted.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(_inline(" ".join(quoted)), st["quote"]))
            continue

        # Lists (bulleted or numbered), including wrapped continuation lines
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            items: list[ListItem] = []
            numbered = bool(re.match(r"^\s*\d+\.\s+", line))
            while i < len(lines) and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]):
                text = re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].strip() and \
                        not re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]) and \
                        lines[i].startswith((" ", "\t")):
                    text += " " + lines[i].strip()
                    i += 1
                items.append(ListItem(Paragraph(_inline(text), st["body"]),
                                      leftIndent=16))
            story.append(ListFlowable(
                items, bulletType="1" if numbered else "bullet",
                start="1" if numbered else None, leftIndent=16,
                bulletFontSize=8, spaceAfter=6,
            ))
            continue

        # Paragraph: gather until a blank line or a new block construct
        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and \
                not re.match(r"^\s*(#{1,6}\s|[-*+]\s|\d+\.\s|>|\||```|---)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(_inline(" ".join(para)), st["body"]))

    doc.build(story)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="Path to the Markdown file")
    ap.add_argument("--out", default=None,
                    help="Output PDF (defaults to the source path with .pdf)")
    ap.add_argument("--title", default=None,
                    help="PDF document title (defaults to the first H1)")
    args = ap.parse_args()

    src = Path(args.source)
    md = src.read_text()
    out = Path(args.out) if args.out else src.with_suffix(".pdf")

    title = args.title
    if title is None:
        m = re.search(r"^#\s+(.*)$", md, re.MULTILINE)
        title = m.group(1).strip() if m else src.stem.replace("_", " ")

    build(md, out, title)
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
