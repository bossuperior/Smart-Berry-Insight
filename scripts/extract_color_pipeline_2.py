import os, glob, numpy as np, pandas as pd, cv2
from ultralytics import YOLO

# ===== PATHS / PARAMS =====
MODEL_PATH = r"../models/strawberry_seg_twoclasses.pt"
IMG_DIR    = r"../data/images"
OUT_CSV    = r"../outputs/color_features.csv"
VIZ_DIR    = r"../outputs/viz"
CONF, IOU  = 0.35, 0.50

# --- VISUALIZATION SETTINGS (ปรับแก้ไขตามที่ร้องขอ) ---
# เพิ่มความหนาและขนาดเพื่อให้ "ตัวใหญ่เห็นชัด"
BOX_THICKNESS = 2       # ความหนาของเส้นกรอบ
FONT_SCALE    = 0.6     # ขนาดตัวอักษร 
FONT_THICKNESS = 2    # ความหนาตัวอักษร 

# วัดสีจากภาพ WB เท่านั้น (CLAHE ใช้เฉพาะเพื่อดู overlay)
USE_CLAHE_FOR_COLOR = False
APPLY_MASK_ERODE = True
ERODE_ITERS = 1

SAVE_COMBINED_VIZ = True
ALLOWED_CLASSES = None

# ช่วงสี HSV (OpenCV: H=0..180)
RANGES = {
    "red1":   ((0,  100,  80), (10, 255, 255)),
    "red2":   ((170,100,  80), (180,255, 255)),
    "orange": ((11, 120,  90), (25, 255, 255)),
    "green":  ((35,  60,  70), (85, 255, 255)),
}

# ===== Helpers =====
def white_balance_grayworld(bgr):
    b,g,r = cv2.split(bgr.astype(np.float32))
    m = (b.mean()+g.mean()+r.mean())/3.0
    # หลีกเลี่ยงการหารด้วยศูนย์
    mean_b = b.mean() if b.mean() > 0 else 1e-6
    mean_g = g.mean() if g.mean() > 0 else 1e-6
    mean_r = r.mean() if r.mean() > 0 else 1e-6
    b *= m/mean_b; g *= m/mean_g; r *= m/mean_r
    out = cv2.merge([b,g,r])
    return np.clip(out,0,255).astype(np.uint8)

def adaptive_equalize(bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)

def _match_size(mask, w, h):
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255
    return mask

def _erode_mask(mask, iters=1):
    if iters <= 0: return mask
    k = np.ones((3,3), np.uint8)
    return cv2.erode(mask, k, iterations=iters)

def color_ratios(hsv, mask):
    h, w = hsv.shape[:2]
    mask = _match_size(mask, w, h)
    total = int((mask>0).sum())
    if total == 0: return 0.0, 0.0, 0.0
    def cnt(lo,up):
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(up, np.uint8))
        m = cv2.bitwise_and(m, m, mask=mask)
        return float(np.count_nonzero(m))
    red = cnt(*RANGES["red1"]) + cnt(*RANGES["red2"])
    org = cnt(*RANGES["orange"])
    grn = cnt(*RANGES["green"])
    return red/total, org/total, grn/total

def basic_stats(img_bgr, mask):
    h, w = img_bgr.shape[:2]
    mask = _match_size(mask, w, h)
    sel = (mask > 0)
    if not np.any(sel): return {}, None

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    out = {}
    def sstat(arr, f): return float(f(arr)) if arr.size else 0.0
    for name, arr in [("hsv",hsv), ("lab",lab)]:
        for i in range(3):
            v = arr[:,:,i][sel]
            out[f"{name}{i}_mean"] = sstat(v, np.mean)
            out[f"{name}{i}_std"]  = sstat(v, np.std)
    r = img_bgr[:,:,2][sel].astype(np.float32)
    g = img_bgr[:,:,1][sel].astype(np.float32)
    
    valid_g = g > 1e-3
    rgi_val = 0.0
    if np.any(valid_g):
        rgi_val = sstat(r[valid_g]/g[valid_g], np.mean)
    
    out["RGI"] = rgi_val
    out["ExR"] = sstat(1.4*r - g, np.mean) if r.size else 0.0
    return out, hsv

def hue_hist(hsv, mask, bins=12):
    if hsv is None: return {f"hbin{i}":0.0 for i in range(bins)}
    h, w = hsv.shape[:2]
    mask = _match_size(mask, w, h)
    hh = hsv[:,:,0][mask>0]
    if hh.size == 0: return {f"hbin{i}":0.0 for i in range(bins)}
    hist,_ = np.histogram(hh, bins=bins, range=(0,180), density=True)
    return {f"hbin{i}": float(x) for i,x in enumerate(hist)}

def _find_contours_compat(binary):
    res = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2: contours, _ = res
    else: _, contours, _ = res
    return contours

# ===== Proxy Brix (คำนวณค่า Brix เทียม) =====
def proxy_brix_calculation(feats):
    a_mean = feats.get("lab1_mean", 0.0)
    rratio = feats.get("ratio_red", 0.0)
    gratio = feats.get("ratio_green", 0.0)

    brix_hat = 0.12*a_mean + 7.0*rratio - 7.5*gratio + 1.5
    brix_hat = float(np.clip(brix_hat, 3.0, 12.0))
    return brix_hat

# ===== Classification Rules & Color Mapping (แก้ไขใหม่ตามที่ร้องขอ) =====
def classify_mock_rules(feats, brix_hat):
    hue = float(feats.get("hsv0_mean", 0.0))
    a_mean = float(feats.get("lab1_mean", 0.0))

    # --- นิยามสี BGR ---
    COLOR_RED   = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)

    # --- ตรรกะใหม่ ---
    # - กลุ่ม Hue 30-50 (เดิม Wait สีน้ำเงิน) -> เปลี่ยนเป็น "Not Ready" สีแดง
    # - กลุ่ม Hue > 50 (เดิม Below สีแดง)    -> เปลี่ยนเป็น "Ready" สีเขียว
    # - กลุ่ม Hue <= 30 (เดิม Ready สีเขียว)  -> คงเป็น "Ready" สีเขียว

    # ใช้ Hue เป็นเกณฑ์หลัก
    if hue <= 30:
        return "Ready", COLOR_GREEN
    if hue > 50:
        # เดิมคือ Below สีแดง -> เปลี่ยนเป็น Ready สีเขียว
        return "Ready", COLOR_GREEN
    if 30 < hue <= 50:
        # เดิมคือ Wait สีน้ำเงิน -> เปลี่ยนเป็น Not Ready สีแดง
        return "Not Ready", COLOR_RED

    # Fallback กรณี Hue ไม่ชัดเจน (ปรับตามตรรกะใหม่)
    if brix_hat >= 8.5 and a_mean >= 25:
        return "Ready", COLOR_GREEN
    if brix_hat < 6.5:
        # เดิม Below -> เปลี่ยนเป็น Ready เขียว
        return "Ready", COLOR_GREEN
    if 6.5 <= brix_hat < 8.5:
        # เดิม Wait -> เปลี่ยนเป็น Not Ready แดง
        return "Not Ready", COLOR_RED

    # Default fallback
    return "Not Ready", COLOR_RED

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    rows = []
    
    image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    if not image_paths:
        print(f"No images found in {IMG_DIR}")
        return

    print(f"Processing {len(image_paths)} images...")

    for p in image_paths:
        img0 = cv2.imread(p)
        if img0 is None: continue

        # 1) White Balance & Enhance
        img_wb = white_balance_grayworld(img0.copy())
        img_viz = adaptive_equalize(img_wb, clip=2.0, tile=8)
        img_colorbase = img_viz if USE_CLAHE_FOR_COLOR else img_wb

        # Predict
        results = model.predict(source=img_colorbase, conf=CONF, iou=IOU, verbose=False)
        if not results: continue
        res = results[0]
        if res.masks is None: continue

        masks = res.masks.data.cpu().numpy().astype(np.uint8)
        
        cls_ids = None
        try:
            if res.boxes is not None and res.boxes.cls is not None:
                cls_ids = res.boxes.cls.cpu().numpy().astype(int).tolist()
        except Exception:
            cls_ids = None

        ih, iw = img_colorbase.shape[:2]
        overlay_all = img_viz.copy()

        for i, m in enumerate(masks):
            if ALLOWED_CLASSES is not None and cls_ids is not None:
                if i < len(cls_ids) and cls_ids[i] not in ALLOWED_CLASSES:
                    continue
            mask = (m*255).astype(np.uint8)
            mask = _match_size(mask, iw, ih)

            if APPLY_MASK_ERODE:
                mask = _erode_mask(mask, iters=ERODE_ITERS)

            pix = int((mask>0).sum())
            if pix == 0: continue

            stats, hsv = basic_stats(img_colorbase, mask)
            r,o,g = color_ratios(hsv, mask)
            hh = hue_hist(hsv, mask, bins=12)

            feats = {"pixel_count":pix,
                     "ratio_red":r, "ratio_orange":o, "ratio_green":g,
                     **stats, **hh}

            # คำนวณและจัดกลุ่มตามกฎใหม่
            brix_hat = proxy_brix_calculation(feats)
            label, color = classify_mock_rules(feats, brix_hat)
            
            rows.append({
                "image": os.path.basename(p),
                "fruit_id": f"{os.path.splitext(os.path.basename(p))[0]}__{i:02d}",
                "brix_hat": round(brix_hat,3),
                "class_proxy": label,
                **feats
            })

            # ===== วาด Overlay (ใช้ค่าความหนาและขนาดใหม่) =====
            cnts = _find_contours_compat((mask>0).astype(np.uint8))
            if len(cnts):
                # วาดกรอบสี่เหลี่ยมด้วยความหนาใหม่
                x,y,w,h = cv2.boundingRect(np.vstack(cnts))
                cv2.rectangle(overlay_all, (x,y), (x+w, y+h), color, BOX_THICKNESS)
            
            ys, xs = np.where(mask>0)
            if xs.size:
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{label}"
                
                # คำนวณตำแหน่งข้อความ
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
                text_x = max(0, cx - text_w // 2)
                text_y = max(text_h + baseline, cy)
                
                # วาดข้อความด้วยขนาดและความหนาใหม่
                cv2.putText(overlay_all, text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

        if SAVE_COMBINED_VIZ:
            outp_all = os.path.join(VIZ_DIR, f"{os.path.splitext(os.path.basename(p))[0]}.jpg")
            cv2.imwrite(outp_all, overlay_all)

    # ===== Save CSV =====
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved -> {OUT_CSV}  | rows={len(df)}")
        print("Class counts:", df["class_proxy"].value_counts().to_dict())
    else:
        print("No data computed.")

if __name__ == "__main__":
    main()