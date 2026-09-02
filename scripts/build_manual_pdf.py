from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REGULAR_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
MONO_FONT = Path("/System/Library/Fonts/Supplemental/Courier New.ttf")


def _register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("SmithSans", str(REGULAR_FONT)))
    pdfmetrics.registerFont(TTFont("SmithSansBold", str(BOLD_FONT)))
    pdfmetrics.registerFont(TTFont("SmithMono", str(MONO_FONT)))


def _plain(text: str) -> str:
    return (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("‑", "-")
        .replace("−", "-")
    )


def _inline(text: str) -> str:
    parts = _plain(text).split(chr(96))
    rendered: list[str] = []
    for index, part in enumerate(parts):
        value = escape(part)
        if index % 2:
            rendered.append(f'<font name="SmithMono" size="8.4">{value}</font>')
        else:
            value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
            rendered.append(value)
    return "".join(rendered)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    normal = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="SmithSans",
        fontSize=9.3,
        leading=13.2,
        textColor=colors.HexColor("#20242A"),
        spaceAfter=5,
    )
    return {
        "normal": normal,
        "title": ParagraphStyle(
            "Title",
            parent=normal,
            fontName="SmithSansBold",
            fontSize=23,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#173B57"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=normal,
            fontName="SmithSansBold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#173B57"),
            spaceBefore=11,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=normal,
            fontName="SmithSansBold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#316A82"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=normal,
            leftIndent=14,
            firstLineIndent=-7,
            bulletIndent=6,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=normal,
            fontName="SmithMono",
            fontSize=7.6,
            leading=10.2,
            leftIndent=7,
            rightIndent=7,
            borderColor=colors.HexColor("#CAD4DA"),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor("#F4F7F8"),
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=normal,
            fontSize=7.8,
            leading=10,
        ),
    }


def _table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        del rows[1]
    cells = [[Paragraph(_inline(cell), styles["table"]) for cell in row] for row in rows]
    count = max(len(row) for row in rows)
    widths = {
        2: [62 * mm, 105 * mm],
        3: [42 * mm, 42 * mm, 83 * mm],
        4: [38 * mm, 30 * mm, 24 * mm, 75 * mm],
    }.get(count, [167 * mm / count] * count)
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#173B57")),
                ("FONTNAME", (0, 0), (-1, 0), "SmithSansBold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9FB3BE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _story(markdown: str):
    styles = _styles()
    story = []
    paragraph: list[str] = []
    lines = markdown.splitlines()
    fence = chr(96) * 3
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(_inline(" ".join(paragraph)), styles["normal"]))
            paragraph.clear()

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith(fence):
            flush_paragraph()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code.append(_plain(lines[index]))
                index += 1
            story.append(Preformatted("\n".join(code), styles["code"]))
        elif stripped.startswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(_table(table_lines, styles))
            story.append(Spacer(1, 7))
            continue
        elif stripped.startswith("# "):
            flush_paragraph()
            if story:
                story.append(PageBreak())
            story.append(Paragraph(_inline(stripped[2:]), styles["title"]))
            story.append(Paragraph("SMITH/SONIC standalone distribution", styles["h2"]))
            story.append(Spacer(1, 5))
        elif stripped.startswith("## "):
            flush_paragraph()
            story.append(CondPageBreak(45 * mm))
            story.append(Paragraph(_inline(stripped[3:]), styles["h1"]))
        elif stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_inline(stripped[4:]), styles["h2"]))
        elif stripped.startswith("- "):
            flush_paragraph()
            bullet_lines = [stripped[2:]]
            next_index = index + 1
            while next_index < len(lines):
                continuation = lines[next_index].strip()
                if (
                    not continuation
                    or continuation.startswith((fence, "|", "#", "- "))
                ):
                    break
                bullet_lines.append(continuation)
                next_index += 1
            story.append(
                KeepTogether(
                    [
                        Paragraph(
                            _inline(" ".join(bullet_lines)),
                            styles["bullet"],
                            bulletText="•",
                        )
                    ]
                )
            )
            index = next_index - 1
        elif not stripped:
            flush_paragraph()
        else:
            paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return story


def _page(canvas, document) -> None:
    canvas.saveState()
    width, _height = A4
    canvas.setStrokeColor(colors.HexColor("#B8C7CE"))
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)
    canvas.setFont("SmithSans", 7.5)
    canvas.setFillColor(colors.HexColor("#60727C"))
    canvas.drawString(20 * mm, 10.5 * mm, "Standalone SMITH / SONIC manual")
    canvas.drawRightString(width - 20 * mm, 10.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_manual(source: Path, output: Path) -> Path:
    _register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=21 * mm,
        title="Standalone SMITH / SONIC Manual",
        author="Vincenzo Barone",
    )
    document.build(_story(source.read_text(encoding="utf-8")), onFirstPage=_page, onLaterPages=_page)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the standalone SMITH manual PDF")
    parser.add_argument("source", type=Path, nargs="?", default=Path("standalone/MANUAL.md"))
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("output/pdf/SMITH_Standalone_Manual.pdf"),
    )
    args = parser.parse_args()
    print(build_manual(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
