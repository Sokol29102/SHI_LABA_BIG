from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
target = ROOT/'deliverables/Lab_Defense_6slides_UA.pdf'
pdf = canvas.Canvas(str(target), pagesize=(960, 540))
pdf.setTitle('CIFAR-10: ML, CNN і Transformer - 6 слайдів')
for index in range(1, 7):
    image = ROOT/f'build/presentation/slide-{index}.png'
    pdf.drawImage(ImageReader(str(image)), 0, 0, width=960, height=540)
    pdf.showPage()
pdf.save()
print('Six-slide visual PDF:', target)
