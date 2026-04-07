"""
Family Photo Sorter — Zero Storage Waste Edition
================================================
- Every photo written to disk EXACTLY ONCE  (hard links = 0 extra bytes)
- Unknown faces → Unknown/ folder
- No face → No_Face/ folder  
- Corrupt files → Corrupt_Files/ folder
- Generates _report.html — browse results in your browser
- Generates _database.json — full metadata for every photo

Usage:
    python sort_photos.py --input "D:/Photos/Unsorted" --output "D:/Photos/Sorted"
    python sort_photos.py --input "D:/Photos/Unsorted" --output "D:/Photos/Sorted" --confidence 0.50
    python sort_photos.py --input "D:/Photos/Unsorted" --output "D:/Photos/Sorted" --dry-run
"""

import os, sys, json, pickle, shutil, argparse, warnings, io, hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFile
from PIL.ExifTags import TAGS
from tqdm import tqdm

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ══════════════════════════════════════════════════════════════════════════════
#  FACENET ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class BasicConv2d(nn.Module):
    def __init__(self, ip, op, ks, st, pd=0):
        super().__init__()
        self.conv = nn.Conv2d(ip, op, ks, stride=st, padding=pd, bias=False)
        self.bn   = nn.BatchNorm2d(op, eps=0.001)
    def forward(self, x): return F.relu(self.bn(self.conv(x)), inplace=True)

class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.b0 = BasicConv2d(256,32,1,1)
        self.b1 = nn.Sequential(BasicConv2d(256,32,1,1), BasicConv2d(32,32,3,1,1))
        self.b2 = nn.Sequential(BasicConv2d(256,32,1,1), BasicConv2d(32,32,3,1,1), BasicConv2d(32,32,3,1,1))
        self.cv = nn.Conv2d(96,256,1); self.rl = nn.ReLU(inplace=False)
    def forward(self, x):
        return self.rl(x + self.scale*self.cv(torch.cat([self.b0(x),self.b1(x),self.b2(x)],1)))

class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.b0 = BasicConv2d(896,128,1,1)
        self.b1 = nn.Sequential(BasicConv2d(896,128,1,1), BasicConv2d(128,128,(1,7),1,(0,3)), BasicConv2d(128,128,(7,1),1,(3,0)))
        self.cv = nn.Conv2d(256,896,1); self.rl = nn.ReLU(inplace=False)
    def forward(self, x):
        return self.rl(x + self.scale*self.cv(torch.cat([self.b0(x),self.b1(x)],1)))

class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()
        self.scale = scale; self.noReLU = noReLU
        self.b0 = BasicConv2d(1792,192,1,1)
        self.b1 = nn.Sequential(BasicConv2d(1792,192,1,1), BasicConv2d(192,192,(1,3),1,(0,1)), BasicConv2d(192,192,(3,1),1,(1,0)))
        self.cv = nn.Conv2d(384,1792,1)
        if not noReLU: self.rl = nn.ReLU(inplace=False)
    def forward(self, x):
        o = x + self.scale*self.cv(torch.cat([self.b0(x),self.b1(x)],1))
        return o if self.noReLU else self.rl(o)

class Mixed6a(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0 = BasicConv2d(256,384,3,2)
        self.b1 = nn.Sequential(BasicConv2d(256,192,1,1), BasicConv2d(192,192,3,1,1), BasicConv2d(192,256,3,2))
        self.b2 = nn.MaxPool2d(3,stride=2)
    def forward(self, x): return torch.cat([self.b0(x),self.b1(x),self.b2(x)],1)

class Mixed7a(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0 = nn.Sequential(BasicConv2d(896,256,1,1), BasicConv2d(256,384,3,2))
        self.b1 = nn.Sequential(BasicConv2d(896,256,1,1), BasicConv2d(256,256,3,2))
        self.b2 = nn.Sequential(BasicConv2d(896,256,1,1), BasicConv2d(256,256,3,1,1), BasicConv2d(256,256,3,2))
        self.b3 = nn.MaxPool2d(3,stride=2)
    def forward(self, x): return torch.cat([self.b0(x),self.b1(x),self.b2(x),self.b3(x)],1)

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1a=BasicConv2d(3,32,3,2);   self.c2a=BasicConv2d(32,32,3,1)
        self.c2b=BasicConv2d(32,64,3,1,1); self.mp3=nn.MaxPool2d(3,stride=2)
        self.c3b=BasicConv2d(64,80,1,1);  self.c4a=BasicConv2d(80,192,3,1)
        self.c4b=BasicConv2d(192,256,3,2)
        self.r1=nn.Sequential(*[Block35(0.17) for _ in range(5)])
        self.m6=Mixed6a()
        self.r2=nn.Sequential(*[Block17(0.1) for _ in range(10)])
        self.m7=Mixed7a()
        self.r3=nn.Sequential(*[Block8(0.2) for _ in range(5)])
        self.b8=Block8(noReLU=True)
        self.ap=nn.AdaptiveAvgPool2d(1); self.do=nn.Dropout(0.6)
        self.ll=nn.Linear(1792,512,bias=False)
        self.bn=nn.BatchNorm1d(512,eps=0.001,momentum=0.1,affine=True)
    def forward(self, x):
        for l in [self.c1a,self.c2a,self.c2b,self.mp3,self.c3b,self.c4a,self.c4b,
                  self.r1,self.m6,self.r2,self.m7,self.r3,self.b8,self.ap,self.do]:
            x = l(x)
        return F.normalize(self.bn(self.ll(x.flatten(1))), p=2, dim=1)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = Path("face_model")
IMG_EXTS   = {".jpg",".jpeg",".png",".webp",".bmp",".tiff",".heic"}

CASCADE_FILES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
]

# ══════════════════════════════════════════════════════════════════════════════
#  HARD LINK UTILITY  (the storage-saving core)
# ══════════════════════════════════════════════════════════════════════════════

def make_hard_link(src: Path, dst: Path):
    """
    Create a hard link at dst pointing to src.
    Hard links:
      - Use ZERO extra disk space (same inode, two directory entries)
      - Work on Windows without admin rights (unlike symlinks)
      - Never break (unlike symlinks when source is moved)
      - Appear as normal files in Explorer / Finder
    Falls back to copy if hard links aren't supported
    (e.g. cross-device, FAT32 USB drives).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Resolve name collision
    final = dst
    n = 1
    while final.exists():
        final = dst.parent / f"{dst.stem}_{n}{dst.suffix}"
        n += 1
    try:
        os.link(str(src), str(final))   # hard link — zero bytes on disk
    except (OSError, NotImplementedError):
        shutil.copy2(str(src), str(final))  # fallback: real copy
    return final

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def safe_load_rgb(img_path):
    """
    Load image as clean RGB numpy array.
    Fixes: Invalid SOS parameters, truncated files,
           corrupt headers, wrong EXIF rotation.
    """
    try:
        pil = Image.open(str(img_path))
        try: pil = ImageOps.exif_transpose(pil)
        except: pass
        pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95, optimize=False)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))
    except Exception:
        return None


def file_hash(path):
    """MD5 hash of file — used to detect true duplicates in input."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_photo_date(path):
    try:
        exif = Image.open(path)._getexif()
        if exif:
            for tid, val in exif.items():
                if TAGS.get(tid) == "DateTimeOriginal":
                    return datetime.strptime(val,"%Y:%m:%d %H:%M:%S").strftime("%Y-%m")
    except: pass
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = FaceNet().eval().to(device)
    w   = MODELS_DIR / "facenet_vggface2.pt"
    if not w.exists():
        print("Downloading FaceNet weights (~90MB)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt",
            str(w)
        )
    state = torch.load(str(w), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    state    = {k.replace("module.",""): v for k,v in state.items()}
    filtered = {k: v for k,v in state.items() if k in net.state_dict()}
    net.load_state_dict(filtered, strict=False)

    with open(MODELS_DIR/"classifier.pkl",    "rb") as f: clf = pickle.load(f)
    with open(MODELS_DIR/"label_encoder.pkl", "rb") as f: le  = pickle.load(f)
    with open(MODELS_DIR/"config.json")            as f: cfg = json.load(f)

    cascades = []
    for cf in CASCADE_FILES:
        p = MODELS_DIR / cf
        if p.exists():
            cascades.append(cv2.CascadeClassifier(str(p)))
    if not cascades:
        raise FileNotFoundError("No cascade XML files found in face_model/")

    print(f"✓ Device   : {device}")
    print(f"✓ People   : {cfg['people']}")
    print(f"✓ Cascades : {len(cascades)} loaded")
    return net, clf, le, cfg, cascades, device

# ══════════════════════════════════════════════════════════════════════════════
#  FACE DETECTION + EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def detect_faces(img_rgb, cascades):
    if img_rgb is None: return []
    h, w   = img_rgb.shape[:2]
    scale  = min(1.0, 1280/max(h,w))
    small  = cv2.resize(img_rgb,(int(w*scale),int(h*scale))) if scale<1 else img_rgb
    gray   = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    eq     = cv2.equalizeHist(gray)

    raw = []
    for cas in cascades:
        for (sf,mn,ms) in [(1.10,5,(60,60)),(1.05,3,(40,40))]:
            boxes = cas.detectMultiScale(eq,scaleFactor=sf,minNeighbors=mn,
                                         minSize=ms,flags=cv2.CASCADE_SCALE_IMAGE)
            if len(boxes)>0:
                for (x,y,bw,bh) in boxes:
                    raw.append((int(x/scale),int(y/scale),int(bw/scale),int(bh/scale)))

    if not raw: return []

    # NMS
    arr   = np.array([[x,y,x+bw,y+bh] for (x,y,bw,bh) in raw], dtype=np.float32)
    areas = (arr[:,2]-arr[:,0])*(arr[:,3]-arr[:,1])
    order = areas.argsort()[::-1]; keep=[]
    while len(order)>0:
        i=order[0]; keep.append(i)
        if len(order)==1: break
        rest=order[1:]
        ix1=np.maximum(arr[i,0],arr[rest,0]); iy1=np.maximum(arr[i,1],arr[rest,1])
        ix2=np.minimum(arr[i,2],arr[rest,2]); iy2=np.minimum(arr[i,3],arr[rest,3])
        inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou=inter/(areas[i]+areas[rest]-inter+1e-6)
        order=rest[iou<0.4]

    H,W=img_rgb.shape[:2]
    result=[]
    for i in keep:
        x1,y1,x2,y2=arr[i]; result.append((int(x1),int(y1),int(x2-x1),int(y2-y1)))
    return result


def embed(img_rgb, x, y, fw, fh, net, device, margin=0.28):
    H,W=img_rgb.shape[:2]; m=int(max(fw,fh)*margin)
    x1=max(0,x-m); y1=max(0,y-m); x2=min(W,x+fw+m); y2=min(H,y+fh+m)
    crop = cv2.resize(img_rgb[y1:y2,x1:x2],(160,160))
    t    = torch.from_numpy((crop.astype(np.float32)-127.5)/128.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        return net(t).squeeze().cpu().numpy()


def classify(emb, clf, le, threshold):
    probs = clf.predict_proba([emb])[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    name  = le.inverse_transform([idx])[0]
    return (name, conf) if conf >= threshold else ("Unknown", conf)

# ══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(output_dir: Path, database: list, stats: dict, people: list):
    """
    Generates a clean HTML report you can open in any browser.
    Shows stats, per-person counts, and a gallery preview.
    """
    total   = sum(stats.values())
    named   = {k:v for k,v in stats.items() if k not in ("Unknown","No_Face","Corrupt_Files")}
    rows = ""
    for r in database[:200]:
        face_strs = "<br>".join(
            f"{f['name']} ({f['confidence']:.0%})" for f in r["faces"]
        )
        rows += (
            f"<tr>"
            f"<td>{r['original_name']}</td>"
            f"<td>{r['folder']}</td>"
            f"<td>{face_strs}</td>"
            f"<td>{r['date']}</td>"
            f"</tr>"
        )
    person_cards = "".join(
        f"""<div class="card">
              <div class="name">{p}</div>
              <div class="count">{stats.get(p,0)} photos</div>
            </div>"""
        for p in sorted(people)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Photo Sort Report</title>
<style>
  body {{font-family:system-ui,sans-serif;margin:0;background:#f5f5f5;color:#222}}
  .header {{background:#1a1a2e;color:#fff;padding:32px 40px}}
  .header h1 {{margin:0 0 6px;font-size:28px}}
  .header p  {{margin:0;opacity:.7;font-size:14px}}
  .section {{padding:28px 40px}}
  .stats {{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:28px}}
  .stat  {{background:#fff;border-radius:12px;padding:20px 28px;min-width:130px;
            box-shadow:0 1px 4px rgba(0,0,0,.08)}}
  .stat .num {{font-size:36px;font-weight:700;color:#1a1a2e}}
  .stat .lbl {{font-size:13px;color:#888;margin-top:4px}}
  .cards {{display:flex;flex-wrap:wrap;gap:14px;margin-top:16px}}
  .card  {{background:#fff;border-radius:10px;padding:16px 22px;
            box-shadow:0 1px 4px rgba(0,0,0,.08);min-width:120px}}
  .card .name  {{font-weight:600;font-size:15px;color:#1a1a2e}}
  .card .count {{font-size:13px;color:#888;margin-top:3px}}
  table  {{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;
            overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
  th     {{background:#1a1a2e;color:#fff;padding:12px 16px;text-align:left;font-size:13px}}
  td     {{padding:10px 16px;border-bottom:1px solid #f0f0f0;font-size:13px}}
  tr:last-child td {{border:none}}
  .badge {{display:inline-block;padding:2px 10px;border-radius:20px;
            font-size:11px;font-weight:600;background:#e8f4fd;color:#1a6fa8}}
  .unknown {{background:#fff3e0;color:#e65100}}
  h2 {{font-size:18px;font-weight:600;color:#1a1a2e;margin:0 0 16px}}
  .storage-note {{background:#e8f5e9;border-left:4px solid #43a047;
                   padding:14px 20px;border-radius:0 8px 8px 0;
                   font-size:14px;margin-bottom:24px}}
</style>
</head>
<body>
<div class="header">
  <h1>📸 Family Photo Sort — Report</h1>
  <p>Generated {datetime.now().strftime("%d %b %Y, %H:%M")}</p>
</div>

<div class="section">
  <div class="storage-note">
    💡 <strong>Zero storage waste:</strong> Group photos are stored once in their combined folder.
    Each person's individual folder contains hard links — they appear as real files but use
    <strong>zero extra disk space</strong>.
  </div>

  <div class="stats">
    <div class="stat"><div class="num">{total}</div><div class="lbl">Total photos</div></div>
    <div class="stat"><div class="num">{sum(named.values())}</div><div class="lbl">Recognised</div></div>
    <div class="stat"><div class="num">{stats.get("Unknown",0)}</div><div class="lbl">Unknown faces</div></div>
    <div class="stat"><div class="num">{stats.get("No_Face",0)}</div><div class="lbl">No face</div></div>
    <div class="stat"><div class="num">{stats.get("Corrupt_Files",0)}</div><div class="lbl">Corrupt files</div></div>
    <div class="stat"><div class="num">{len(people)}</div><div class="lbl">People</div></div>
  </div>

  <h2>People</h2>
  <div class="cards">{person_cards}</div>
</div>

<div class="section">
  <h2>Photo log (first 200)</h2>
  <table>
    <tr><th>File</th><th>Sorted into</th><th>Faces detected</th><th>Date</th></tr>
    {rows}
  </table>
</div>
</body>
</html>"""

    report_path = output_dir / "_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SORT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def sort_photos(input_dir, output_dir, confidence=None, dry_run=False):
    input_dir  = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    net, clf, le, cfg, cascades, device = load_models()
    threshold = confidence if confidence is not None else cfg["confidence_thresh"]

    print(f"\nInput          : {input_dir}")
    print(f"Output         : {output_dir}")
    print(f"Threshold      : {threshold}")
    print(f"Dry run        : {dry_run}")
    print(f"Storage method : hard links (zero extra bytes for group photos)\n")

    # ── Collect all images ────────────────────────────────────────────────────
    all_imgs = sorted([
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    ])
    print(f"Found {len(all_imgs)} images\n")

    # ── Deduplicate by hash before processing ─────────────────────────────────
    print("Checking for duplicate files in input...")
    seen_hashes = {}
    unique_imgs = []
    dupes       = 0
    for p in tqdm(all_imgs, desc="Deduplicating", unit="file"):
        h = file_hash(p)
        if h not in seen_hashes:
            seen_hashes[h] = p
            unique_imgs.append(p)
        else:
            dupes += 1
    print(f"  Unique files : {len(unique_imgs)}")
    print(f"  Duplicates   : {dupes} (skipped)\n")

    stats    = defaultdict(int)
    database = []

    for img_path in tqdm(unique_imgs, desc="Sorting", unit="img"):

        # 1. Load image safely (fixes Invalid SOS + all JPEG corruption)
        img_rgb = safe_load_rgb(img_path)

        # 2. Completely unreadable → Corrupt_Files/
        if img_rgb is None:
            stats["Corrupt_Files"] += 1
            if not dry_run:
                make_hard_link(img_path, output_dir/"Corrupt_Files"/img_path.name)
            database.append({
                "original_name": img_path.name,
                "original_path": str(img_path),
                "folder"       : "Corrupt_Files",
                "faces"        : [],
                "date"         : "unknown"
            })
            continue

        # 3. Detect all faces
        boxes = detect_faces(img_rgb, cascades)

        # 4. No face → No_Face/
        if not boxes:
            stats["No_Face"] += 1
            if not dry_run:
                make_hard_link(img_path, output_dir/"No_Face"/img_path.name)
            database.append({
                "original_name": img_path.name,
                "original_path": str(img_path),
                "folder"       : "No_Face",
                "faces"        : [],
                "date"         : get_photo_date(img_path)
            })
            continue

        # 5. Classify every detected face
        face_results = []
        for (x,y,fw,fh) in boxes:
            emb  = embed(img_rgb, x, y, fw, fh, net, device)
            name, conf = classify(emb, clf, le, threshold)
            face_results.append({"name": name, "confidence": round(conf, 3)})

        # Deduplicate people (same person detected twice in one photo)
        seen_names = []
        for r in face_results:
            if r["name"] not in seen_names:
                seen_names.append(r["name"])

        # 6. Determine folder name
        known   = [n for n in seen_names if n != "Unknown"]
        has_unk = "Unknown" in seen_names

        if not known and has_unk:
            folder = "Unknown"
        elif known and not has_unk:
            folder = "+".join(sorted(known))
        elif known and has_unk:
            folder = "+".join(sorted(known)) + "+Unknown"
        else:
            folder = "Unknown"

        stats[folder] += 1

        if not dry_run:
            # ── STORAGE-EFFICIENT PLACEMENT ───────────────────────────────
            #
            # Rule:  every photo is written to disk ONCE.
            #
            # Case A — solo photo (one person):
            #   Hard link directly into Person/ folder.
            #   Cost: 1 × file size on disk.
            #
            # Case B — group photo (2+ people, e.g. "Rahul+Priya"):
            #   Real file goes into the combined "Rahul+Priya/" folder.
            #   Hard link placed in each person's folder.
            #   Cost: still 1 × file size on disk total.
            #   Both Rahul/ and Priya/ show the photo. Zero duplication.
            #
            # Case C — Unknown / No_Face / Corrupt:
            #   Hard link into the respective special folder.
            #   Cost: 1 × file size.
            #
            # Total disk usage = sum of original file sizes.  Nothing more.
            # ─────────────────────────────────────────────────────────────

            # Master copy (or first hard link) goes to the folder folder
            master = make_hard_link(img_path, output_dir / folder / img_path.name)

            # For group photos: hard link into each individual's folder too
            if len(known) > 1:
                for person in known:
                    make_hard_link(img_path, output_dir / person / img_path.name)
                # Also hard link Unknown's copy to Unknown/ if mixed
                if has_unk:
                    make_hard_link(img_path, output_dir / "Unknown" / img_path.name)

        database.append({
            "original_name": img_path.name,
            "original_path": str(img_path),
            "folder"       : folder,
            "faces"        : face_results,
            "date"         : get_photo_date(img_path)
        })

    # ── Save database ─────────────────────────────────────────────────────────
    if not dry_run:
        db_path = output_dir / "_database.json"
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(database, f, indent=2, ensure_ascii=False)

        # ── Generate HTML report ──────────────────────────────────────────────
        people      = cfg["people"]
        report_path = generate_report(output_dir, database, stats, people)

    # ── Print summary ─────────────────────────────────────────────────────────
    total = len(unique_imgs)
    sep   = "═" * 52
    print(f"\n{sep}")
    print("  SORTING COMPLETE")
    print(sep)
    print(f"  Total photos processed : {total}")
    print(f"  Duplicates skipped     : {dupes}")
    print(f"  Corrupt files          : {stats.get('Corrupt_Files',0)}")
    print(f"  No face detected       : {stats.get('No_Face',0)}")
    print(f"  Unknown faces          : {stats.get('Unknown',0)}")
    print()
    print("  Named folders:")
    skip = {"Corrupt_Files","No_Face","Unknown"}
    for folder, count in sorted(stats.items(), key=lambda x:-x[1]):
        if folder not in skip:
            tag = "  (hard links in individual folders too)" if "+" in folder else ""
            print(f"    {folder:40s} {count:5d} photos{tag}")
    print()
    print("  Special folders:")
    for f in ["Unknown","No_Face","Corrupt_Files"]:
        if stats.get(f,0):
            print(f"    {f:40s} {stats[f]:5d}")

    if not dry_run:
        print(f"\n  Database  → {output_dir}/_database.json")
        print(f"  Report    → {output_dir}/_report.html")
        print(f"              (open this in your browser)")
    print(sep)

    unknown_pct = stats.get("Unknown",0) / max(total,1)
    if unknown_pct > 0.20:
        print(f"\n  ⚠ {unknown_pct:.0%} of photos went to Unknown/")
        print(f"  Try: python sort_photos.py ... --confidence 0.45")

    if dry_run:
        print("\n  DRY RUN — no files were written.")
        print("  Remove --dry-run to actually sort.")

# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sort family photos by face recognition — zero storage waste"
    )
    ap.add_argument("--input",      required=True,            help="Folder with unsorted photos")
    ap.add_argument("--output",     required=True,            help="Where to write sorted output")
    ap.add_argument("--confidence", type=float, default=None, help="0.0–1.0 (default: from config)")
    ap.add_argument("--dry-run",    action="store_true",      help="Preview without writing any files")
    a = ap.parse_args()
    sort_photos(a.input, a.output, a.confidence, a.dry_run)