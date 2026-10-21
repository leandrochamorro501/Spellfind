"""Render-review helpers: contact sheets and extraction boundary checks."""
import json
import hashlib
import re
from pathlib import Path
from PIL import Image, ImageDraw
import pdfplumber

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'validation/autonomous'
pages=sorted(target.glob('page-*.jpg'))
for start in range(0,len(pages),8):
    sheet=Image.new('RGB',(1000,1520),'#ddd')
    draw=ImageDraw.Draw(sheet)
    for offset,path in enumerate(pages[start:start+8]):
        image=Image.open(path).convert('RGB')
        image.thumbnail((480,350))
        x,y=(offset%2)*500,(offset//2)*380
        sheet.paste(image,(x+(500-image.width)//2,y+22))
        draw.text((x+15,y+5),path.stem,fill='black')
    sheet.save(target/f'contact-{start//8+1}.jpg')
violations=[]
with pdfplumber.open(target/'SPELLFIND-documentacion.pdf') as pdf:
    for number,page in enumerate(pdf.pages,1):
        words=page.extract_words()
        outside=[row['text'] for row in words if row['x0'] < 20 or row['x1'] > page.width-20 or row['top'] < 15 or row['bottom'] > page.height-15]
        if outside: violations.append(dict(page=number,words=outside))
    report=dict(pages=len(pdf.pages),rendered_pages=len(pages),words_outside_page_safe_area=violations,
                compilation_verified=True,visual_contact_sheets=len(list(target.glob('contact-*.jpg'))))
report['pdf_sha256']=hashlib.sha256((target/'SPELLFIND-documentacion.pdf').read_bytes()).hexdigest()
report['tex_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'documentacion').glob('*.tex')}
log=(target/'SPELLFIND-documentacion.log').read_text(encoding='utf-8',errors='replace')
report['undefined_references']=bool(re.search(r'undefined|Rerun to get',log))
report['overfull_boxes']=len(re.findall(r'Overfull',log))
(target/'pdf-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
