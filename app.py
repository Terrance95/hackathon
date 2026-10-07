import streamlit as st
import forensics as F

st.set_page_config(page_title="Image Tampering Detection", layout="wide", page_icon="🔍")

# Inject Custom CSS for dark gradient & modern typography
st.markdown("""
<style>
    /* Main App Gradient Background */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #1a2332 100%);
    }
    
    /* Elegant Title Styling */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00e5ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Image Tampering Detection")
st.caption("Advanced Digital Image Forensics & AI Analysis Platform")
st.info(F.DISCLAIMER)

up = st.file_uploader("Upload an Image to Analyze", type=["jpg", "jpeg", "png"], help="Max size 10MB")

def safe(label, fn, default=None):
    try: return fn()
    except Exception as e:
        st.warning(f"{label} failed: {e}"); return default

if up:
    data = up.getvalue()
    try:
        img, info = F.load_image(data, up.name)
    except ValueError as e:
        st.error(str(e)); st.stop()

    st.markdown("---")
    
    # 1. Background computation
    with st.spinner("🔍 Extracting Features & Executing Error Level Analysis..."):
        # ELA
        ela_res, ela_arr = {}, None
        for q in (90, 95, 98):
            r = safe(f"ELA Q{q}", lambda q=q: F.ela(img, q))
            if r:
                arr, im, s = r; ela_res[q] = {"stats": s, "image": im}
                if q == 95: ela_arr = arr
                
        # Features & Meta
        feats = safe("Features", lambda: F.features(img), {})
        meta = safe("Metadata", lambda: F.metadata(img), {"exif_available": False, "useful": {}, "count": 0})
        
        # Region Localization
        boxes, heat, ov, bx = [], None, None, None
        if ela_arr is not None:
            loc = safe("Localization", lambda: F.localize(ela_arr, img))
            if loc: heat, ov, bx, boxes = loc
            
        # ML Inference
        bundle, reason = F.load_model()
        pred, perr = None, reason
        if bundle and feats and ela_res:
            pred, perr = F.predict(bundle, feats, ela_res.get(95, next(iter(ela_res.values())))['stats'])
        ev = F.evidence({q: v["stats"] for q, v in ela_res.items()}, feats, meta, pred, boxes, perr) if feats and ela_res else []
        v = F.verdict(pred)

    # 2. Hero Section: Final Verdict
    st.subheader("System Verdict")
    if pred:
        is_real = "Real" in v
        color = "normal" if is_real else "inverse" # Streamlit metric delta color
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Prediction", f"{pred['label']}", delta="Authentic" if is_real else "Anomalies Detected", delta_color=color)
        c2.metric("Confidence Score", f"{pred['confidence']:.2%}" if pred['confidence'] else "N/A", delta="AI Assisted", delta_color="off")
        c3.metric("Suspicious Regions", f"{len(boxes)}", delta="Found via ELA" if boxes else "Clean", delta_color="inverse" if boxes else "normal")
    else:
        st.warning("ML classification unavailable — trained model not found." if reason else f"ML classification unavailable — {perr}.")

    st.markdown("---")
    
    # 3. Tabbed UI for Clear Separation
    t_viz, t_ela, t_data, t_rep = st.tabs([
        "👁️ Visual Analysis", 
        "🔬 Error Level Analysis (ELA)", 
        "📊 Features & Meta", 
        "📄 Forensic Report"
    ])
    
    with t_viz:
        st.markdown("### Side-by-Side Comparison")
        st.caption("Compare the original image with AI-highlighted anomaly regions to clearly tell the difference.")
        cv1, cv2 = st.columns(2)
        cv1.image(img, caption="Original Uploaded Image", use_column_width="always")
        if ov is not None:
            cv2.image(ov, caption="Heatmap Overlay (Anomalies Highlighted)", use_column_width="always")
        
        if bx is not None:
            st.markdown("### Detected Bounding Boxes")
            st.image(bx, caption="Potentially suspicious regions based on high error levels", use_column_width="always")
            if boxes:
                st.dataframe(boxes, hide_index=True, use_container_width=True)
            else:
                st.info("No concentrated anomaly regions exceeded the threshold.")
    
    with t_ela:
        st.markdown("### Error Level Analysis")
        st.caption("Higher brightness in ELA indicates regions that were recently edited or not uniformly compressed.")
        cols = st.columns(3)
        for col, (q, dat) in zip(cols, ela_res.items()):
            col.image(dat["image"], caption=f"ELA Artifacts (JPEG Quality {q})", use_column_width="always")
            
    with t_data:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("### Image Features")
            if feats:
                st.dataframe({k: [round(val, 4) if isinstance(val, float) else val] for k, val in feats.items()}, hide_index=True, use_container_width=True)
        with d2:
            st.markdown("### EXIF Metadata")
            st.write(f"Metadata Status: **{'Found' if meta['exif_available'] else 'Missing'}**")
            if meta["useful"]: 
                st.table(meta["useful"])
            else: 
                st.write("No useful metadata available")
                
    with t_rep:
        st.markdown("### Full Forensic Evidence")
        for e in ev: st.markdown(f"- {e}")
        
        st.markdown("### Download Data")
        ela_stats_only = {q: dat["stats"] for q, dat in ela_res.items()}
        report_txt = F.report(info, ela_stats_only, feats, meta, pred, perr, boxes, ev, v)
        st.download_button(
            label="📄 Download Full Forensic Report (.txt)", 
            data=report_txt,
            file_name=f"report_{up.name}.txt",
            use_container_width=True
        )
