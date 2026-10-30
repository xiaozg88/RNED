#!/usr/bin/env python3
"""Convert a simple Markdown manuscript to a minimal .docx (zipfile + XML only).

Supports: #/##/### headings, **bold**, - lists, 1. lists, | tables, paragraphs,
and embedded images via ![caption](path). Image captions are rendered as a
centered paragraph below the picture; a paragraph line starting with ``**表``
or ``**图`` is rendered as a centered caption. No external dependencies.
"""
import re
import struct
import sys
import zipfile
import os
from html import escape
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_omml import to_omml

EMU_PER_INCH = 914400
MAX_W_IN = 6.0
MAX_H_IN = 8.0


def runs(text):
    """Split text on **bold** and {{red|...}} into styled runs."""
    out = []
    for segment in re.split(r"(\{\{red\|.*?\}\})", text):
        if not segment:
            continue
        match = re.fullmatch(r"\{\{red\|(.*)\}\}", segment)
        content = match.group(1) if match else segment
        for i, part in enumerate(re.split(r"\*\*", content)):
            if part:
                out.append((part, i % 2 == 1, bool(match)))
    return out


def text_runs(text):
    """Plain text runs honoring **bold** and red markup."""
    return "".join(
        '<w:r><w:rPr>' + ('<w:b/>' if b else '') + ('<w:color w:val="FF0000"/>' if red else '') + '</w:rPr>'
        f'<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'
        for t, b, red in runs(text)
    )


def inline_body(text):
    """Interleave text runs and inline math ($...$) OMML."""
    if "$" not in text:
        return text_runs(text)
    parts = text.split("$")
    out = []
    for idx, part in enumerate(parts):
        if idx % 2 == 0:
            out.append(text_runs(part))
        else:
            out.append(to_omml(part))
    return "".join(out)


def _p(body, align=None, style=None):
    ppr_parts = []
    if style:
        ppr_parts.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr_parts.append(f'<w:jc w:val="{align}"/>')
    ppr = f"<w:pPr>{''.join(ppr_parts)}</w:pPr>" if ppr_parts else ""
    return f"<w:p>{ppr}{body}</w:p>"


def code_line(text):
    """Monospace paragraph for fenced code blocks."""
    body = (
        '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
        f'<w:sz w:val="18"/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    )
    ppr = '<w:pPr><w:spacing w:before="0" w:after="0"/><w:ind w:left="360"/></w:pPr>'
    return f"<w:p>{ppr}{body}</w:p>"


def para(text, style=None, align=None):
    body = inline_body(text)
    return _p(body, align=align, style=style)


def display_equation(latex, number):
    omml = to_omml(latex)
    ppr = (
        '<w:pPr>'
        '<w:tabs><w:tab w:val="center" w:pos="4680"/><w:tab w:val="right" w:pos="9360"/></w:tabs>'
        '<w:spacing w:before="120" w:after="120"/>'
        '<w:jc w:val="center"/>'
        '</w:pPr>'
    )
    return (
        f'<w:p>{ppr}<w:r><w:tab/></w:r>{omml}'
        f'<w:r><w:tab/></w:r>'
        f'<w:r><w:t xml:space="preserve">({number})</w:t></w:r></w:p>'
    )


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def _image_xml(rel_id, doc_pr_id, cx, cy, name):
    return (
        f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{doc_pr_id}" name="{name}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{doc_pr_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
    )


def table(rows):
    ncols = max((len(r) for r in rows), default=0)
    grid = "".join('<w:gridCol w:w="1000"/>' for _ in range(ncols))
    xml = ["<w:tbl>", '<w:tblPr><w:tblStyle w:val="TableGrid"/></w:tblPr>', f"<w:tblGrid>{grid}</w:tblGrid>"]
    for r in rows:
        xml.append("<w:tr>")
        for cell in r:
            xml.append(f"<w:tc>{para(cell)}</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def split_table_row(line):
    """Split a Markdown row while preserving {{red|...}} as one cell fragment."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells = []
    cell = ""
    index = 0
    while index < len(line):
        if line.startswith("{{red|", index):
            end = line.find("}}", index)
            if end >= 0:
                cell += line[index : end + 2]
                index = end + 2
                continue
        if line[index] == "|":
            cells.append(cell.strip())
            cell = ""
        else:
            cell += line[index]
        index += 1
    cells.append(cell.strip())
    return cells


class Doc:
    def __init__(self):
        self.body = []
        self.images = []  # (path, rel_id, doc_pr_id, cx, cy, name)
        self._img_counter = 0
        self.eq_no = 0

    def add_image(self, path, caption):
        path = Path(path)
        w, h = png_size(path)
        scale = min(MAX_W_IN / w, MAX_H_IN / h)
        cx = int(w * scale * EMU_PER_INCH)
        cy = int(h * scale * EMU_PER_INCH)
        self._img_counter += 1
        rel_id = f"rIdImg{self._img_counter}"
        name = f"Picture {self._img_counter}"
        self.images.append((path, rel_id, self._img_counter, cx, cy, name))
        xml = _p(_image_xml(rel_id, self._img_counter, cx, cy, name), align="center")
        self.body.append(xml)
        if caption.strip():
            m = re.match(r"^(图\s*\d+|表\s*\d+|Figure\s*\d+|Table\s*\d+)\s*(.*)$", caption.strip())
            if m:
                cap = f"**{m.group(1)}** {m.group(2)}".rstrip()
            else:
                cap = caption.strip()
            self.body.append(para(cap, align="center"))

    def render(self):
        return "".join(self.body)


def convert(md, doc):
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                doc.body.append(code_line(lines[i].rstrip()))
                i += 1
            i += 1  # skip closing fence
            continue
        if line.lstrip().startswith("$$"):
            content = line.strip()
            if content.endswith("$$") and len(content) >= 4:
                latex = content[2:-2]
                doc.eq_no += 1
                doc.body.append(display_equation(latex, doc.eq_no))
                i += 1
                continue
        if line.lstrip().startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = split_table_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    tbl.append(cells)
                i += 1
            doc.body.append(table(tbl))
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if m:
            doc.add_image(m.group(2), m.group(1))
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            style = {1: "Heading1", 2: "Heading2", 3: "Heading3", 4: "Heading4"}.get(lvl, "Heading4")
            doc.body.append(para(m.group(2), style))
            i += 1
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            doc.body.append(para("\u2022 " + m.group(1), "ListParagraph"))
            i += 1
            continue
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            doc.body.append(para(m.group(1), "ListParagraph"))
            i += 1
            continue
        txt = line.strip()
        if re.match(r"^\*\*(图|表|Figure|Table)\s*\d", txt):
            doc.body.append(para(txt, align="center"))
        else:
            doc.body.append(para(txt))
        i += 1


def ct_with_png():
    return CT.replace(
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>',
    )


CT = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''

PKG_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

DOC_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/><w:pPr><w:spacing w:before="140" w:after="70"/></w:pPr><w:rPr><w:rFonts w:eastAsia="SimHei"/><w:b/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:pPr><w:ind w:left="420"/></w:pPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''

DOC = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" w:header="851" w:footer="992" w:gutter="0"/></w:sectPr></w:body></w:document>'''


def main():
    src, dst = sys.argv[1], sys.argv[2]
    md = open(src, encoding="utf-8").read()
    doc = Doc()
    convert(md, doc)

    # document.xml.rels with image relationships
    rel_extra = "".join(
        f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{i}.png"/>'
        for _path, rel_id, i, _cx, _cy, _name in doc.images
    )
    doc_rels = DOC_RELS.replace("</Relationships>", rel_extra + "</Relationships>")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct_with_png())
        z.writestr("_rels/.rels", PKG_RELS)
        z.writestr("word/document.xml", DOC.format(body=doc.render()))
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/styles.xml", STYLES)
        for path, _rel_id, i, _cx, _cy, _name in doc.images:
            z.write(str(path), f"word/media/image{i}.png")
    print(f"wrote {dst} ({len(doc.images)} images)")


if __name__ == "__main__":
    main()
