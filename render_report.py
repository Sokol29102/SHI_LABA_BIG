"""Render the verified report source with embedded Ukrainian fonts."""
import json
from pathlib import Path
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'deliverables'


def main():
    for name, filename in [('Arial', 'arial.ttf'), ('ArialBold', 'arialbd.ttf'), ('ArialItalic', 'ariali.ttf'), ('ArialBoldItalic', 'arialbi.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts')/filename)))
    pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='ArialBold', italic='ArialItalic', boldItalic='ArialBoldItalic')
    body = ParagraphStyle('BodyUA', fontName='Arial', fontSize=10.6, leading=14.5, textColor=colors.HexColor('#1f2937'), spaceAfter=7, allowWidows=0, allowOrphans=0)
    heading = ParagraphStyle('HeadingUA', parent=body, fontName='ArialBold', fontSize=18, leading=23, spaceAfter=14)
    subhead = ParagraphStyle('SubheadingUA', parent=body, fontName='ArialBold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=6)
    cell = ParagraphStyle('CellUA', parent=body, fontSize=9.0, leading=12.4, spaceAfter=0)
    header = ParagraphStyle('HeaderCellUA', parent=cell, fontName='ArialBold', textColor=colors.white)
    caption = ParagraphStyle('CaptionUA', parent=body, fontName='ArialItalic', fontSize=8.7, leading=12, textColor=colors.HexColor('#475569'), spaceAfter=10)
    pages = json.loads((OUT/'report_content.json').read_text(encoding='utf-8'))
    story = []
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(escape(page['title']), heading))
        for block in page['blocks']:
            kind = block['type']
            if kind == 'p':
                story.append(Paragraph(escape(block['text']), body))
            elif kind == 'h':
                story.append(Paragraph(escape(block['text']), subhead))
            elif kind == 'ref':
                story.append(Paragraph(escape(block['text'])+f' <a href="{escape(block["url"], quote=True)}" color="#1d4ed8">Першоджерело</a>', body))
            elif kind == 'table':
                rows = [[Paragraph(escape(str(value)).replace('\n', '<br/>'), header if r == 0 else cell) for value in row]
                        for r, row in enumerate(block['rows'])]
                widths = block.get('widths')
                table = Table(rows, colWidths=[value*cm for value in widths] if widths else None, repeatRows=1, hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17324d')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                    ('TOPPADDING', (0, 0), (-1, -1), 5.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5),
                    ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#17324d')),
                    ('LINEBELOW', (0, 1), (-1, -1), .35, colors.HexColor('#dbe3ec')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f8fb')])]))
                story.extend([table, Spacer(1, 10)])
            elif kind == 'image':
                image = Image(str(OUT/'assets'/block['file']))
                original_width, original_height = image.imageWidth, image.imageHeight
                width = block.get('width_cm', 16.7)*cm
                image.drawWidth = width
                image.drawHeight = original_height*width/original_width
                image.hAlign = 'LEFT'
                story.append(KeepTogether([image, Spacer(1, 5), Paragraph(escape(block['caption']), caption)]))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#dbe3ec'))
        canvas.line(1.9*cm, 1.55*cm, A4[0]-1.9*cm, 1.55*cm)
        canvas.setFont('Arial', 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(1.9*cm, 1.12*cm, 'CIFAR-10 | ML, CNN, Transformer | 11.11.2026')
        canvas.drawRightString(A4[0]-1.9*cm, 1.12*cm, str(doc.page))
        canvas.restoreState()
    document = SimpleDocTemplate(str(OUT/'Lab_Report_UA.pdf'), pagesize=A4,
        rightMargin=1.9*cm, leftMargin=1.9*cm, topMargin=1.8*cm, bottomMargin=2.0*cm,
        title='Порівняння ML, CNN та Transformer на CIFAR-10', author='')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print('PDF report rendered:', OUT/'Lab_Report_UA.pdf')


if __name__ == '__main__':
    main()
