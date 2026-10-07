import io, os, glob, json, datetime
import numpy as np, cv2
from PIL import Image, ImageChops, ExifTags

DISCLAIMER = "This tool provides forensic indicators and should not be treated as definitive proof of image manipulation."
ALLOWED = {"JPEG", "PNG"}

def load_image(data: bytes, name: str):
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data)); img.load()
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image: {e}")
    if img.format not in ALLOWED:
        raise ValueError(f"Unsupported format: {img.format}")
    info = {"Filename": name, "Format": img.format, "Dimensions": f"{img.width} x {img.height}",
            "File size": f"{len(data)/1024:.1f} KB"}
    return img, info

def ela(img, quality):
    rgb = img.convert("RGB"); buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality); buf.seek(0)
    diff = ImageChops.difference(rgb, Image.open(buf).convert("RGB"))
    arr = np.asarray(diff).astype(np.float32)
    mx = max(arr.max(), 1.0)
    return arr, Image.fromarray(np.clip(arr * (255.0 / mx), 0, 255).astype(np.uint8)), \
        {"mean": float(arr.mean()), "max": float(arr.max()), "std": float(arr.std())}

def features(img):
    g = np.asarray(img.convert("L"))
    edges = cv2.Canny(g, 100, 200)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    hist = np.bincount(g.ravel(), minlength=256) / g.size
    h = hist[hist > 0]
    return {"mean_brightness": float(g.mean()), "std_dev": float(g.std()),
            "edge_density": float((edges > 0).mean()), "laplacian_variance": float(lap.var()),
            "texture_mean_abs_gradient": float(np.abs(np.gradient(g.astype(float))[0]).mean()),
            "entropy": float(-(h * np.log2(h)).sum()), "width": img.width, "height": img.height}

def metadata(img):
    out = {}
    try:
        for k, v in (img.getexif() or {}).items():
            out[ExifTags.TAGS.get(k, str(k))] = str(v)[:100]
        try:
            for k, v in img.getexif().get_ifd(0x8769).items():
                out[ExifTags.TAGS.get(k, str(k))] = str(v)[:100]
        except Exception: pass
    except Exception: pass
    keys = ["Make", "Model", "DateTime", "DateTimeOriginal", "Software"]
    useful = {k: out[k] for k in keys if k in out}
    return {"exif_available": bool(out), "useful": useful, "count": len(out)}

def load_model(folder="models"):
    """Looks for model/scaler/feature-name files in `folder`. Returns dict or None + reason."""
    try:
        files = glob.glob(os.path.join(folder, "*.pkl")) + glob.glob(os.path.join(folder, "*.joblib"))
        if not files: return None, "trained model not found"
        import joblib
        mf = next((f for f in files if "scaler" not in f.lower() and "feature" not in f.lower()), None)
        if not mf: return None, "trained model not found"
        sf = next((f for f in files if "scaler" in f.lower()), None)
        ff = next((f for f in files + glob.glob(os.path.join(folder, "*.json")) if "feature" in f.lower()), None)
        names = None
        if ff:
            names = json.load(open(ff)) if ff.endswith(".json") else list(joblib.load(ff))
        return {"model": joblib.load(mf), "scaler": joblib.load(sf) if sf else None,
                "names": names, "file": os.path.basename(mf)}, None
    except Exception as e:
        return None, f"model failed to load ({e})"

def predict(bundle, feats, ela_stats):
    allf = {**feats, **{f"ela_{k}": v for k, v in ela_stats.items()}}
    try:
        names = bundle["names"] or list(feats.keys())
        X = np.array([[allf[n] for n in names]], dtype=float)
    except KeyError as e:
        return None, f"feature mismatch: missing {e}"
    m = bundle["model"]; exp = getattr(m, "n_features_in_", X.shape[1])
    if exp != X.shape[1]: return None, f"feature mismatch: model expects {exp}, got {X.shape[1]}"
    try:
        if bundle["scaler"] is not None: X = bundle["scaler"].transform(X)
        pred = m.predict(X)[0]
        conf = float(np.max(m.predict_proba(X))) if hasattr(m, "predict_proba") else None
        cls = getattr(m, "classes_", None)
        s = str(pred).lower()
        tampered = s in ("1", "tampered", "fake", "forged", "tp", "true") or (s not in ("0", "real", "authentic", "au", "false") and bool(pred))
        return {"label": "Likely Tampered" if tampered else "Likely Real", "raw": str(pred), "confidence": conf}, None
    except Exception as e:
        return None, f"prediction failed ({e})"

def localize(arr, img, thr_pct=97, min_area_frac=0.002):
    g = arr.mean(axis=2)
    g = cv2.GaussianBlur(g, (0, 0), max(1, min(img.size) / 200))
    norm = (255 * g / max(g.max(), 1e-6)).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)[:, :, ::-1]
    t = np.percentile(g, thr_pct)
    mask = (g >= max(t, 1e-6)).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    base = np.asarray(img.convert("RGB")).copy(); area_min = min_area_frac * g.size
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= area_min: boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area_pct": round(100 * a / g.size, 2)})
    boxes.sort(key=lambda b: -b["area_pct"]); boxes = boxes[:5]
    for b in boxes: cv2.rectangle(base, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (255, 0, 0), max(2, img.width // 300))
    ov = cv2.addWeighted(np.asarray(img.convert("RGB")), 0.6, heat, 0.4, 0)
    return Image.fromarray(heat), Image.fromarray(ov), Image.fromarray(base), boxes

def evidence(ela_res, feats, meta, pred, boxes, pred_err):
    ev = []
    for q, s in ela_res.items():
        ev.append(f"ELA Q{q}: mean error {s['mean']:.2f}, max {s['max']:.0f}, std {s['std']:.2f} (indicator only, not proof).")
    ev.append(f"Brightness {feats['mean_brightness']:.1f}, std dev {feats['std_dev']:.1f}, entropy {feats['entropy']:.2f} bits.")
    ev.append(f"Edge density {feats['edge_density']:.3f}; Laplacian variance {feats['laplacian_variance']:.1f}.")
    ev.append("Metadata: " + (", ".join(f"{k}={v}" for k, v in meta["useful"].items()) if meta["useful"] else "No useful metadata available (not evidence of manipulation)."))
    ev.append(f"ML: {pred['label']} (raw={pred['raw']}" + (f", confidence={pred['confidence']:.2%})" if pred['confidence'] is not None else ")") if pred
              else f"ML classification unavailable — {pred_err}.")
    ev.append(f"Localization: {len(boxes)} potentially suspicious region(s) found from high ELA response." if boxes
              else "Localization: no region exceeded the ELA anomaly threshold.")
    return ev

def verdict(pred):
    return pred["label"] if pred else "Insufficient evidence — ML model unavailable"

def report(info, ela_res, feats, meta, pred, pred_err, boxes, ev, v):
    L = ["IMAGE TAMPERING DETECTION - FORENSIC REPORT", f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "",
         "== Image ==", *[f"{k}: {x}" for k, x in info.items()], "", "== ELA ==",
         *[f"Quality {q}: " + ", ".join(f"{k}={x:.3f}" for k, x in s.items()) for q, s in ela_res.items()], "",
         "== Features ==", *[f"{k}: {x:.4f}" if isinstance(x, float) else f"{k}: {x}" for k, x in feats.items()], "",
         "== Metadata ==", f"EXIF available: {meta['exif_available']}",
         *([f"{k}: {x}" for k, x in meta['useful'].items()] or ["No useful metadata available"]), "",
         "== ML ==", (f"{pred['label']} (raw={pred['raw']}, confidence={pred['confidence']})" if pred else f"ML classification unavailable - {pred_err}"), "",
         "== Potentially suspicious regions ==", *([str(b) for b in boxes] or ["None detected"]), "",
         "== Evidence ==", *[f"- {e}" for e in ev], "", f"VERDICT: {v}", "", "Disclaimer: " + DISCLAIMER]
    return "\n".join(L)
