import os, glob, numpy as np, pandas as pd, cv2
from ultralytics import YOLO

# ===== PATHS / PARAMS =====
MODEL_PATH = r"../models/strawberry_seg_twoclasses.pt"
IMG_DIR    = r"../data/images"
OUT_CSV    = r"../outputs/color_features.csv"
VIZ_DIR    = r"../outputs/viz"
CONF, IOU  = 0.35, 0.50

# วัดสีจากภาพ WB เท่านั้น (CLAHE ใช้เฉพาะเพื่อดู overlay)
USE_CLAHE_FOR_COLOR = False
APPLY_MASK_ERODE = True           # หด mask เพื่อกันขอบปนเปื้อน
ERODE_ITERS = 1

# แสดงผลรวมหลายผลในภาพเดียว (แทนการเซฟทีละผล)
SAVE_COMBINED_VIZ = True          # เซฟภาพรวมผลลัพธ์หลายวัตถุพร้อมกัน
SAVE_PER_INSTANCE_VIZ = False     # ถ้ายังอยากได้ไฟล์ต่อผล ให้เปิดเป็น True

# ถ้าต้องการกรองเฉพาะคลาสบางรายการ: ใส่เลขคลาสตามโมเดล (e.g., [0] หรือ [0,1])
ALLOWED_CLASSES = None            # None = ไม่กรอง, ใช้ทุกคลาสที่เจอ

# เกณฑ์ (ถ้าต้องการให้ระบบปรับอัตโนมัติจาก distribution ให้เปิด AUTO_CAL)
T_MIN_DEFAULT   = 6.5
T_READY_DEFAULT = 8.5
AUTO_CAL        = False           # ถ้า True: ใช้เปอร์เซ็นไทล์ 30/70 ปรับเกณฑ์ชั่วคราว

# ช่วงสี HSV (OpenCV: H=0..180)
# ยก S,V ขั้นต่ำขึ้น เพื่อตัดแดงหม่น/เงามืดออก
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
    b *= m/(b.mean()+1e-6); g *= m/(g.mean()+1e-6); r *= m/(r.mean()+1e-6)
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
    valid = g > 1e-3
    out["RGI"] = sstat(r[valid]/g[valid], np.mean) if np.any(valid) else 0.0
    out["ExR"] = sstat(1.4*r - g, np.mean) if r.size else 0.0
    return out, hsv

def hue_hist(hsv, mask, bins=12):
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

# ===== Proxy Brix & Classification =====
def proxy_brix_and_class(feats, t_min=T_MIN_DEFAULT, t_ready=T_READY_DEFAULT):
    # สูตรอ่อนลง + clip ช่วง
    a_mean = feats.get("lab1_mean", 0.0)     # a*
    rratio = feats.get("ratio_red", 0.0)
    gratio = feats.get("ratio_green", 0.0)

    brix_hat = 0.12*a_mean + 7.0*rratio - 7.5*gratio + 1.5
    brix_hat = float(np.clip(brix_hat, 3.0, 12.0))

    if brix_hat >= t_ready: label="Ready"
    elif brix_hat <  t_min: label="Below"
    else:                   label="Wait"
    return brix_hat, label

# กฎจัดกลุ่มสถานะเพื่อ mock output ตามโจทย์
# - Wait (สีน้ำเงิน): 6.5 ≤ Brix_hat < 8.5 หรือ 30° < Hue ≤ 50°
# - Ready (สีเขียว):  Brix_hat ≥ 8.5 และ a* ≥ 25 หรือ Hue ≤ 30°
# - Below (สีแดง):   Brix_hat < 6.5 หรือ Hue > 50°
def classify_final(feats, brix_hat,
                   t_min=6.5, t_ready=8.5, a_star_cut=25,
                   use_hue_fallback=True, pix_th=400):
    """
    จัดกลุ่มตามสเปก:
    - Ready: brix_hat >= t_ready และ a* >= a_star_cut
    - Wait : 6.5 <= brix_hat < 8.5
    - Below: brix_hat < 6.5
    ใช้ Hue (OpenCV 0..180) เป็น fallback เฉพาะกรณี pixel_count น้อยมาก
    """
    a_mean = float(feats.get("lab1_mean", 0.0))
    pix    = int(feats.get("pixel_count", 0))
    hue    = float(feats.get("hsv0_mean", 0.0))  # 0..180

    # กฎหลัก
    if brix_hat >= t_ready and a_mean >= a_star_cut:
        label = "Ready"
    elif brix_hat < t_min:
        label = "Below"
    else:
        label = "Wait"  # 6.5 ≤ brix_hat < 8.5

    # Fallback เชิงประคอง (เฉพาะกรณี mask เล็ก)
    if use_hue_fallback and pix < pix_th:
        # mapping คร่าวๆ: 30°≈15(cv2), 50°≈25(cv2)
        if hue <= 15 and label != "Ready":
            label = "Ready"
        elif hue > 25 and label != "Below":
            label = "Below"

    color_map = {"Ready": (0,255,0), "Wait": (255,0,0), "Below": (0,0,255)}
    return label, color_map[label]

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    rows, brix_list = [], []

    for p in sorted(glob.glob(os.path.join(IMG_DIR, "*.*"))):
        img0 = cv2.imread(p)
        if img0 is None: continue

        # 1) ภาพสำหรับ "วัดสี"
        img_wb = white_balance_grayworld(img0.copy())
        if USE_CLAHE_FOR_COLOR:
            img_colorbase = adaptive_equalize(img_wb, clip=2.0, tile=8)
        else:
            img_colorbase = img_wb

        # 2) ภาพสำหรับ "แสดงผล" (อยากคมชัดขึ้น)
        img_viz = adaptive_equalize(img_wb, clip=2.0, tile=8)

        # 3) YOLO predict ที่ภาพ white-balanced (เพื่อขอบ mask ที่สม่ำเสมอ)
        res = model.predict(source=img_colorbase, conf=CONF, iou=IOU, verbose=False)[0]
        if res.masks is None: 
            continue

        masks = res.masks.data.cpu().numpy().astype(np.uint8)  # [N,H,W] 0/1
        # จับคู่คลาสต่อ instance (ถ้ามี และถ้าต้องการกรอง)
        cls_ids = None
        try:
            if res.boxes is not None and res.boxes.cls is not None:
                cls_ids = res.boxes.cls.cpu().numpy().astype(int).tolist()
        except Exception:
            cls_ids = None

        ih, iw = img_colorbase.shape[:2]
        overlay_all = img_viz.copy()  # รวมหลายวัตถุในภาพเดียว

        for i, m in enumerate(masks):
            # กรองคลาสถ้ากำหนดไว้
            if ALLOWED_CLASSES is not None and cls_ids is not None:
                if i < len(cls_ids) and cls_ids[i] not in ALLOWED_CLASSES:
                    continue
            mask = (m*255).astype(np.uint8)
            mask = _match_size(mask, iw, ih)

            # 4) หด mask ตัดขอบรบกวน
            if APPLY_MASK_ERODE:
                mask = _erode_mask(mask, iters=ERODE_ITERS)

            pix = int((mask>0).sum())
            stats, hsv = basic_stats(img_colorbase, mask)
            r,o,g = color_ratios(hsv, mask)
            hh = hue_hist(hsv, mask, bins=12)

            feats = {"pixel_count":pix,
                     "ratio_red":r, "ratio_orange":o, "ratio_green":g,
                     **stats, **hh}

            brix_hat, _ = proxy_brix_and_class(feats, T_MIN_DEFAULT, T_READY_DEFAULT)
            label, color = classify_final(feats, brix_hat, 
                                          t_min=T_MIN_DEFAULT, 
                                          t_ready=T_READY_DEFAULT, 
                                          a_star_cut=25)
            brix_list.append(brix_hat)

            rows.append({
                "image": os.path.basename(p),
                "fruit_id": f"{os.path.splitext(os.path.basename(p))[0]}__{i:02d}",
                "brix_hat": round(brix_hat,3),
                "class_proxy": label,
                "T_min": T_MIN_DEFAULT,
                "T_ready": T_READY_DEFAULT,
                **feats
            })

            # 5) วาด overlay รวมหลายวัตถุในภาพเดียว + ต่อผล (ออปชัน)
            cnts = _find_contours_compat((mask>0).astype(np.uint8))
            # คำนวนกรอบสี่เหลี่ยมล้อมผล เพื่อให้เป็น "กรอบ" ตามคำขอ
            if len(cnts):
                x,y,w,h = cv2.boundingRect(np.vstack(cnts))
                cv2.rectangle(overlay_all, (x,y), (x+w, y+h), color, 2)
            else:
                cv2.drawContours(overlay_all, cnts, -1, color, 2)
            ys, xs = np.where(mask>0)
            if xs.size:
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"#{i+1} Brix~{brix_hat:.1f} {label}"
                cv2.putText(overlay_all, text,
                            (max(0, cx-70), max(15, cy-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # ถ้าต้องการภาพต่อผล
            if SAVE_PER_INSTANCE_VIZ:
                overlay = img_viz.copy()
                if len(cnts):
                    x,y,w,h = cv2.boundingRect(np.vstack(cnts))
                    cv2.rectangle(overlay, (x,y), (x+w, y+h), color, 2)
                else:
                    cv2.drawContours(overlay, cnts, -1, color, 2)
                if xs.size:
                    cv2.putText(overlay, f"Brix~{brix_hat:.1f} {label}",
                                (max(0, cx-60), max(20, cy-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                outp = os.path.join(VIZ_DIR, f"{os.path.splitext(os.path.basename(p))[0]}__{i:02d}.jpg")
                cv2.imwrite(outp, overlay)

        # เซฟภาพรวมของทั้งภาพเดียว
        if SAVE_COMBINED_VIZ:
            outp_all = os.path.join(VIZ_DIR, f"{os.path.splitext(os.path.basename(p))[0]}.jpg")
            cv2.imwrite(outp_all, overlay_all)

    # ===== Auto-calibrate thresholds (ออปชัน) =====
    if AUTO_CAL and brix_list:
        brix_arr = np.array(brix_list, dtype=float)
        t_min   = float(np.percentile(brix_arr, 30))
        t_ready = float(np.percentile(brix_arr, 70))
    
        new_rows = []
        for r in rows:
            # เตรียม feats คืนจาก r
            feats = {k: r[k] for k in r.keys() 
                     if k not in ("image","fruit_id","brix_hat","class_proxy","T_min","T_ready")}
            label, _ = classify_final(feats, r["brix_hat"], 
                                      t_min=t_min, t_ready=t_ready, a_star_cut=25)
            r["class_proxy"] = label
            r["T_min"] = round(t_min,3)
            r["T_ready"] = round(t_ready,3)
            new_rows.append(r)
        rows = new_rows

if __name__ == "__main__":
    main()
