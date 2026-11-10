"""Regenerate the three manuscript figures from the archived descriptive summary."""
import argparse
import json
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#0072B2', '#D55E00', '#00856B']
MODELS = ['M1', 'M2', 'M3']
record = json.loads((ROOT / 'supplementary_materials/summary.json').read_text(encoding='utf-8'))
SHORT = record['protocols']
all_runs = record['all_runs']
scales = record['scales']

def draw_text(c, x, y, s, size=9, *, align='left', bold=False, color='#222222'):
    c.setFillColor(HexColor(color))
    c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
    {'left': c.drawString, 'right': c.drawRightString, 'center': c.drawCentredString}[align](x, y, str(s))


def line(c, x1, y1, x2, y2, color='#D3D9DE', width=.6):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)


def mark(c, x, y, index, color, size=3.3):
    c.setFillColor(HexColor(color))
    if index == 0:
        c.circle(x, y, size, fill=1, stroke=0)
    elif index == 1:
        c.rect(x-size, y-size, 2*size, 2*size, fill=1, stroke=0)
    else:
        p = c.beginPath()
        p.moveTo(x, y+size*1.25)
        p.lineTo(x-size*1.15, y-size)
        p.lineTo(x+size*1.15, y-size)
        p.close()
        c.drawPath(p, fill=1, stroke=0)


def figure_canvas(n, height):
    c = canvas.Canvas(str(FIG / f'Figure_{n}.pdf'), pagesize=(425.2, height), invariant=1)
    c.setAuthor('')
    c.setTitle(f'Figure {n}')
    return c


def save_figure(c, n):
    c.save()
    doc = pdfium.PdfDocument(str(FIG / f'Figure_{n}.pdf'))
    im = doc[0].render(scale=300/72).to_pil().convert('RGB')
    im.save(FIG / f'Figure_{n}.tif', compression='tiff_lzw', dpi=(300, 300))


def render_figures():
    c = figure_canvas(1, 200)
    left, right = 95, 358
    x = lambda value: left + (value-.44) / .21 * (right-left)
    for i in range(6):
        t = .45 + .04*i
        line(c, x(t), 41, x(t), 158)
        draw_text(c, x(t), 26, f'{t:.2f}', align='center', size=8)
    for i, r in enumerate(all_runs):
        y = 145 - 31*i
        draw_text(c, 4, y-3, SHORT[i], size=9)
        line(c, x(r['reference']), y, x(r['validation']), y, '#7A8791', 1.4)
        mark(c, x(r['validation']), y, 0, '#D55E00')
        mark(c, x(r['reference']), y, 1, '#0072B2')
        draw_text(c, 420, y-3, f"{r['gap']:+.4f}", align='right', size=9)
    draw_text(c, 420, 164, 'Signed gap', align='right', size=8, bold=True)
    mark(c, 99, 181, 0, '#D55E00')
    draw_text(c, 107, 178, 'Validation')
    mark(c, 183, 181, 1, '#0072B2')
    draw_text(c, 191, 178, 'Geographic reference')
    draw_text(c, 222, 9, 'Mean intersection over union', align='center')
    save_figure(c, 1)

    c = figure_canvas(2, 230)
    for i, r in enumerate(all_runs):
        x0, y0 = 11 + (i % 2)*212, 125 - (i // 2)*108
        draw_text(c, x0+96, y0+87, SHORT[i], align='center', size=10, bold=True)
        draw_text(c, x0+51, y0+71, 'Validation', align='center', size=8)
        draw_text(c, x0+142, y0+71, 'Reference', align='center', size=8)
        for j in range(3):
            y = y0+55-21*j
            draw_text(c, x0, y-3, j+1, size=8, color='#505861')
            line(c, x0+25, y, x0+171, y)
        for j, model in enumerate(MODELS):
            ya = y0+55-21*r['validation_order'].index(model)
            yb = y0+55-21*r['reference_order'].index(model)
            line(c, x0+51, ya, x0+142, yb, COLORS[j], 1.6)
            mark(c, x0+51, ya, j, COLORS[j])
            mark(c, x0+142, yb, j, COLORS[j])
            draw_text(c, x0+43, ya-3, model, align='right', size=8)
            draw_text(c, x0+150, yb-3, model, size=8)
    save_figure(c, 2)

    c = figure_canvas(3, 222)
    lx, rx, bot, top = 53, 401, 39, 178
    xx = [lx+39, (lx+rx)/2, rx-39]
    y = lambda value: bot + value/.11*(top-bot)
    for i in range(6):
        t = i*.02
        line(c, lx, y(t), rx, y(t))
        draw_text(c, lx-8, y(t)-3, f'{t:.2f}', align='right', size=8)
    for j, values in enumerate(scales.values()):
        for i in range(2):
            line(c, xx[i], y(values[i]), xx[i+1], y(values[i+1]), COLORS[j], 1.7)
        for a, b in zip(xx, values):
            mark(c, a, y(b), j, COLORS[j])
            draw_text(c, a, y(b)+(7 if j == 0 else -12), f'+{b:.4f}', size=8, align='center', color=COLORS[j])
    for a, label in zip(xx, ['0.5r', '1r', '2r']):
        draw_text(c, a, 24, label, align='center')
    draw_text(c, 227, 8, 'Nominal spatial setting', align='center')
    draw_text(c, 52, 185, 'Validation minus geographic-reference mIoU', size=9)
    for j, label in enumerate(['Spatial blocks', 'Buffered blocks']):
        mark(c, 143+121*j, 207, j, COLORS[j])
        draw_text(c, 151+121*j, 204, label, size=9)
    save_figure(c, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'build/figures')
    FIG = parser.parse_args().output_dir
    FIG.mkdir(parents=True, exist_ok=True)
    render_figures()
    print(f'Wrote three PDF and 300 dpi TIFF figures to {FIG}')
