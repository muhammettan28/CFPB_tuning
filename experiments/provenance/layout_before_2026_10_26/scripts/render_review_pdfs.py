#!/usr/bin/env python3
"""Render review PDFs and check word bounds. Visual approval remains a manual step.

Optional QA dependencies: Poppler command-line tools and Pillow.
"""
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw

PAPER=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);report={}
    for stem in ['manuscript','supplementary_material','response_to_reviewers']:
        source=PAPER/(stem+'.pdf');d=out/stem;d.mkdir()
        subprocess.run(['pdftoppm','-scale-to','1400','-png',str(source),str(d/'page')],check=True,stderr=subprocess.DEVNULL)
        subprocess.run(['pdftotext','-layout',str(source),str(d/'text.txt')],check=True)
        subprocess.run(['pdftotext','-bbox',str(source),str(d/'bounds.html')],check=True)
        pages=sorted(d.glob('page-*.png'))
        info=subprocess.check_output(['pdfinfo','-f','1','-l',str(len(pages)),str(source)],text=True)
        rotations={int(n):int(rot) for n,rot in re.findall(r'Page\s+(\d+) rot:\s+(\d+)',info)}
        for start in range(0,len(pages),4):
            sheet=Image.new('RGB',(1720,2480),'#e2e2e2');draw=ImageDraw.Draw(sheet)
            for j,path in enumerate(pages[start:start+4]):
                im=Image.open(path).convert('RGB');im.thumbnail((840,1200))
                x=(j%2)*860+(860-im.width)//2;y=(j//2)*1240+25
                sheet.paste(im,(x,y));draw.text((x,y-18),f'{stem} / {start+j+1}',fill='black')
            sheet.save(d/f'contact-{start//4+1}.png')
        root=ET.parse(d/'bounds.html');ns={'x':'http://www.w3.org/1999/xhtml'};bad=[];rotated=[]
        for i,page in enumerate(root.findall('.//x:page',ns),1):
            w,h=float(page.attrib['width']),float(page.attrib['height'])
            if rotations.get(i,0)%180:
                # Poppler reports portrait box sizes but mixed header/content
                # coordinates for pdflscape; also inspect this rendered page.
                w=h=max(w,h);rotated.append(i)
            for word in page.findall('.//x:word',ns):
                a=word.attrib
                if float(a['xMin']) < -1 or float(a['yMin']) < -1 or float(a['xMax'])>w+1 or float(a['yMax'])>h+1:
                    bad.append((i,''.join(word.itertext()),a))
        report[stem]={'pages':len(pages),'out_of_page_words':bad,'rotated_pages_require_visual_check':rotated,
                      'layout_rendered':True,'visual_review':'pending'}
        print(stem,len(pages),'pages; bounds violations',len(bad),flush=True)
    (out/'LAYOUT_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(r['out_of_page_words'] for r in report.values()):raise SystemExit('Review unexpected word bounds before approval')

if __name__=='__main__':main()
