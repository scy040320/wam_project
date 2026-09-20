"""D13 task-0 online object-slot evidence audit; never reads simulator ground truth."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

SLOTS = {"object_alphabet_soup":(0.18,0.42,0.33,0.72), "object_tomato_sauce":(0.45,0.34,0.64,0.73), "target_basket":(0.64,0.39,0.96,0.90)}
def crop(image, bounds):
    h,w=image.shape[:2]; x1,y1,x2,y2=bounds; return image[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]
def evidence(image,bounds):
    region=crop(image,bounds); hsv=cv2.cvtColor(region,cv2.COLOR_RGB2HSV); gray=cv2.cvtColor(region,cv2.COLOR_RGB2GRAY)
    texture=float(cv2.Laplacian(gray,cv2.CV_64F).var()); saturation=float(hsv[...,1].mean()); luminance=float(gray.mean()); nonblack=float(np.mean(gray>8))
    confidence=min(1.0,0.45*nonblack+0.30*min(texture/140.0,1.0)+0.25*min(saturation/100.0,1.0))
    return {"texture":texture,"saturation":saturation,"luminance":luminance,"nonblack_fraction":nonblack,"confidence":confidence,"state":"asserted" if confidence>=0.45 else "unasserted"}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--images",required=True); parser.add_argument("--out",required=True); args=parser.parse_args()
    source,out=Path(args.images),Path(args.out); out.mkdir(parents=True,exist_ok=True)
    paths=sorted(source.rglob("*current_primary*.png")) or sorted(source.rglob("*.png"))[:30]
    rows=[]
    for index,path in enumerate(paths):
        image=np.asarray(Image.open(path).convert("RGB")); row={"record_image":str(path),"source":"online_primary_image_only"}; preview=Image.fromarray(image).convert("RGB"); draw=ImageDraw.Draw(preview)
        for slot,bounds in SLOTS.items():
            row[slot]=evidence(image,bounds); h,w=image.shape[:2]; x1,y1,x2,y2=bounds
            draw.rectangle((x1*w,y1*h,x2*w,y2*h),outline="lime" if row[slot]["state"]=="asserted" else "orange",width=2); draw.text((x1*w+2,y1*h+2),f"{slot}: {row[slot]['confidence']:.2f}",fill="white")
        preview.save(out/f"audit_{index:03d}.png"); rows.append(row)
    compact=[{"record_image":r["record_image"],"source":r["source"],**{f"{s}_state":r[s]["state"] for s in SLOTS},**{f"{s}_confidence":round(r[s]["confidence"],4) for s in SLOTS}} for r in rows]
    (out/"online_slot_records.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    with (out/"online_slot_audit.csv").open("w",newline="",encoding="utf-8") as h:
        writer=csv.DictWriter(h,fieldnames=list(compact[0])); writer.writeheader(); writer.writerows(compact)
    (out/"method_card.json").write_text(json.dumps({"name":"task0_roi_visual_slot_evidence_v1","uses_simulator_gt":False,"uses_offline_labels":False,"inputs":["online primary camera image","reviewed task ROI spec"],"safety":"low evidence is unasserted, not silently valid","limitation":"task-specific ROI evidence is an engineering bridge, not a general semantic detector"},indent=2),encoding="utf-8")
    print(json.dumps({"images":len(rows),"out":str(out.resolve())}))
if __name__=="__main__": main()
