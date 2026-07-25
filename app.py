import os
import io
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF

import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO

# ==============================================================================
# 1. DYNAMIC CBAM ATTENTION MODULE REGISTRATION
# ==============================================================================
class CBAM(nn.Module):
    def __init__(self, c1=None, *args, kernel_size=7, **kwargs):
        super().__init__()
        self.c1 = c1
        self.kernel_size = kernel_size
        self.init_done = False
        if isinstance(c1, int):
            self._build(c1)

    def _build(self, c1, device=None, dtype=None):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, max(8, c1 // 16), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, c1 // 16), c1, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, self.kernel_size, padding=self.kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        if device:
            self.to(device=device, dtype=dtype)
        self.init_done = True

    def forward(self, x):
        if not self.init_done:
            self._build(x.shape[1], device=x.device, dtype=x.dtype)

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)
        
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp_map = torch.cat([avg_sp, max_sp], dim=1)
        return x * self.sigmoid_spatial(self.conv_spatial(sp_map))

setattr(modules, 'CBAM', CBAM)
setattr(modules.block, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)

# ==============================================================================
# 2. SOFT RETINEX PREPROCESSING PIPELINE
# ==============================================================================
def fast_multi_scale_retinex_416(img, sigma_list=[15, 80, 250]):
    img_float = np.float32(img) + 1.0
    log_img = np.log10(img_float)
    retinex = np.zeros_like(img_float)

    for sigma in sigma_list:
        s = max(3, int(sigma * (416 / 1024)))
        if s % 2 == 0: 
            s += 1
        blur = cv2.GaussianBlur(img_float, (s, s), sigma / 4.0)
        retinex += log_img - np.log10(blur + 1.0)

    retinex = retinex / len(sigma_list)
    return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_soft_blend_retinex(img_bgr, blend_weight=0.5):
    img_resized = cv2.resize(img_bgr, (416, 416), interpolation=cv2.INTER_AREA)
    retinex_img = fast_multi_scale_retinex_416(img_resized)
    blended = cv2.addWeighted(img_resized, 1.0 - blend_weight, retinex_img, blend_weight, 0)
    
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# ==============================================================================
# 3. PDF REPORT GENERATOR ENGINE
# ==============================================================================
def generate_pdf_report(processed_results, summary_counts, model_name):
    pdf = FPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title Banner
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "AI Fabric Quality Inspection Report", ln=True, align="C")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Model Architecture: {model_name}", ln=True, align="C")
    pdf.cell(0, 5, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(8)
    
    # Executive Summary Box
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. Executive Batch Summary", ln=True)
    pdf.set_font("Helvetica", "", 9)
    
    pdf.cell(38, 7, f"Total Samples: {summary_counts['total']}", border=1, align="C")
    pdf.cell(38, 7, f"Grade A (Pass): {summary_counts['grade_a']}", border=1, align="C")
    pdf.cell(38, 7, f"Grade B (Warn): {summary_counts['grade_b']}", border=1, align="C")
    pdf.cell(38, 7, f"Grade C (Reject): {summary_counts['grade_c']}", border=1, align="C")
    pdf.cell(38, 7, f"Total Defects: {summary_counts['total_defects']}", border=1, align="C")
    pdf.ln(12)
    
    # Table Header
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Detailed Inspection Log", ln=True)
    
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 235, 252)
    pdf.cell(12, 8, "#", border=1, fill=True, align="C")
    pdf.cell(75, 8, "Filename", border=1, fill=True)
    pdf.cell(35, 8, "Grade", border=1, fill=True, align="C")
    pdf.cell(25, 8, "Defects", border=1, fill=True, align="C")
    pdf.cell(25, 8, "Latency", border=1, fill=True, align="C")
    pdf.ln()
    
    # Table Content
    pdf.set_font("Helvetica", "", 8)
    for item in processed_results:
        clean_name = item['name'][:38] + "..." if len(item['name']) > 40 else item['name']
        pdf.cell(12, 7, str(item['idx']), border=1, align="C")
        pdf.cell(75, 7, clean_name, border=1)
        pdf.cell(35, 7, item['grade'], border=1, align="C")
        pdf.cell(25, 7, str(item['defects']), border=1, align="C")
        pdf.cell(25, 7, f"{item['latency']:.1f} ms", border=1, align="C")
        pdf.ln()
        
    return bytes(pdf.output())

# ==============================================================================
# 4. LOCAL MODEL PATH MAPPING
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(BASE_DIR, "weight")

ALL_MODEL_PATHS = {
    "YOLOv8n + CBAM (Proposed - Soft Retinex)": os.path.join(WEIGHT_DIR, "yolov8n_cbam_softretinex.pt"),
    "YOLOv8n (Baseline)": os.path.join(WEIGHT_DIR, "yolov8n_baseline.pt"),
    "YOLO11n (Baseline)": os.path.join(WEIGHT_DIR, "yolov11n_baseline.pt"),
    "YOLOv8n (Default Pretrained)": os.path.join(BASE_DIR, "yolov8n.pt")
}

MODEL_PATHS = {name: path for name, path in ALL_MODEL_PATHS.items() if os.path.exists(path)}

if not MODEL_PATHS:
    MODEL_PATHS["YOLOv8n (Default)"] = os.path.join(BASE_DIR, "yolov8n.pt")

# ==============================================================================
# 5. STREAMLIT APPLICATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(page_title="AI Fabric Inspector", page_icon="🧵", layout="wide")

st.markdown("""
    <style>
    .metric-box {
        border: 1px solid #00d2ff;
        border-radius: 6px;
        padding: 10px;
        text-align: center;
        background-color: #0b1426;
        box-shadow: 0 0 8px rgba(0, 210, 255, 0.25);
    }
    .metric-title {
        font-size: 10px;
        color: #8a99ad;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    .metric-value-cyan { font-size: 22px; color: #00d2ff; font-weight: bold; }
    .metric-value-green { font-size: 22px; color: #00ff66; font-weight: bold; }
    .metric-value-orange { font-size: 22px; color: #ffaa00; font-weight: bold; }
    .metric-value-red { font-size: 22px; color: #ff3366; font-weight: bold; }
    .green-code { color: #00ff66; font-family: monospace; font-weight: bold; font-size: 13px; word-break: break-all; }
    
    .tip-box {
        background-color: #0d1e36;
        border-left: 4px solid #00d2ff;
        border-radius: 4px;
        padding: 12px;
        margin-top: 12px;
        margin-bottom: 15px;
        font-size: 12px;
        color: #d1dbe5;
    }
    .tip-header {
        color: #00d2ff;
        font-weight: bold;
        font-size: 13px;
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.title("⚙️ AI Calibration")

selected_model_name = st.sidebar.selectbox("Model Architecture:", list(MODEL_PATHS.keys()))
enable_retinex = st.sidebar.checkbox("Enable Soft Retinex Preprocessing", value=True)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.30, 0.01)
iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.05, 1.0, 0.45, 0.01)

st.sidebar.markdown("""
<div class="tip-box">
    <div class="tip-header">💡 Defect Sensitivity Tips</div>
    <ul style="padding-left: 15px; margin: 0;">
        <li><b>Glide Confidence Left (0.15–0.25):</b> Uncover faint/subtle defects like yarn slubs, minor pilling, or tiny oil spots.</li>
        <li><b>Glide Confidence Right (0.35–0.50):</b> Eliminate false alarms on complex weaving textures.</li>
        <li><b>Soft Retinex Preprocessing:</b> Keep enabled to boost defect contrast under uneven shadows.</li>
        <li><b>NMS IoU Adjustment:</b> Lower threshold if multiple overlapping boxes outline a single defect.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.title("📊 Active Metadata")
meta_placeholder = st.sidebar.empty()

@st.cache_resource
def load_yolo_model(path):
    if path and os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

def generate_cam_heatmap(model, img_pil, alpha=0.4):
    """
    Generates a class activation heatmap (Grad-CAM / EigenCAM equivalent) 
    overlaid on the original fabric image for Explainable AI (XAI).
    """
    orig_np = np.array(img_pil.convert("RGB"))
    h, w, _ = orig_np.shape

    # 1. Resize and prepare image tensor for model input
    img_resized = cv2.resize(orig_np, (640, 640))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    activations = []
    def hook_fn(module, input, output):
        if isinstance(output, (tuple, list)):
            activations.append(output[0])
        else:
            activations.append(output)

    try:
        # Hook into the neck feature-aggregation layer
        py_model = model.model
        target_layer = py_model.model[-2]
        handle = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = py_model(tensor)

        handle.remove()

        if activations:
            act = activations[0]
            # Compute spatial feature intensity (EigenCAM / Activation mapping)
            heatmap = torch.mean(act, dim=1).squeeze().cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            # Resize heatmap to match input fabric size
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            
            # Apply JET color map (Blue = Normal Fabric, Red/Yellow = Defect Attention)
            color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

            # Superimpose overlay on original image
            overlay = cv2.addWeighted(orig_np, 1 - alpha, color_heatmap_rgb, alpha, 0)
            return overlay, color_heatmap_rgb
    except Exception as e:
        print(f"Heatmap error: {e}")
        return orig_np, None

    return orig_np, None

model_path = MODEL_PATHS.get(selected_model_name)
model = load_yolo_model(model_path)

# ------------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# ------------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Drag and Drop Fabric Images (Upload up to 10 Images at once)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    processed_results = []
    grade_a_count = 0
    grade_b_count = 0
    grade_c_count = 0
    total_defects = 0

    if model is None:
        st.error(f"❌ Could not load weights for `{selected_model_name}`.")
    else:
        for idx, uploaded_file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if enable_retinex:
                processed_img_bgr = apply_soft_blend_retinex(img_bgr)
            else:
                processed_img_bgr = cv2.resize(img_bgr, (416, 416))

            t_start = time.time()
            results = model.predict(source=processed_img_bgr, conf=confidence_threshold, iou=iou_threshold, verbose=False)
            latency_ms = (time.time() - t_start) * 1000

            num_detections = len(results[0].boxes)
            total_defects += num_detections

            if num_detections == 0:
                grade = "Grade A (Pass)"
                grade_a_count += 1
            elif num_detections == 1:
                grade = "Grade B (Warn)"
                grade_b_count += 1
            else:
                grade = "Grade C (Reject)"
                grade_c_count += 1

            res_plotted_bgr = results[0].plot()
            res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)

            confidences = [f"{box.conf.item():.2f}" for box in results[0].boxes]
            conf_str = ", ".join(confidences) if confidences else "N/A"

            processed_results.append({
                "idx": idx + 1,
                "name": uploaded_file.name,
                "img_rgb": img_rgb,
                "res_plotted_rgb": res_plotted_rgb,
                "defects": num_detections,
                "grade": grade,
                "latency": latency_ms,
                "resolution": f"{w} x {h} px",
                "confidences": conf_str
            })

    # TOP METRICS BANNER
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

    with mcol1:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Total Samples</div><div class="metric-value-cyan">{len(uploaded_files)}</div></div>', unsafe_allow_html=True)
    with mcol2:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Grade A (Pass)</div><div class="metric-value-green">{grade_a_count}</div></div>', unsafe_allow_html=True)
    with mcol3:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Grade B (Warn)</div><div class="metric-value-orange">{grade_b_count}</div></div>', unsafe_allow_html=True)
    with mcol4:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Grade C (Reject)</div><div class="metric-value-red">{grade_c_count}</div></div>', unsafe_allow_html=True)
    with mcol5:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Total Defects</div><div class="metric-value-cyan">{total_defects}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    active_item = processed_results[0]
    meta_placeholder.markdown(f"""
    **Filename:**  
    <span class="green-code">{active_item['name']}</span>  
    
    **Resolution:** `{active_item['resolution']}`  
    
    **AI Latency:** <span class="green-code">{active_item['latency']:.1f} ms</span>  
    
    **Defects Detected:** `{active_item['defects']}`
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # UI TABS
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ All Samples Gallery (Visual Grid)", 
        "🔍 Single Sample Inspection", 
        "📋 Quality Audit Trail & CSV/PDF",
        "📊 Research Benchmarks"
    ])

    # --- TAB 1: VISUAL GRID ---
    with tab1:
        st.subheader("🖼️ Batch Inspection Grid (All Uploaded Images)")
        st.caption("All processed fabric samples rendered simultaneously with neural bounding boxes.")

        for i in range(0, len(processed_results), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(processed_results):
                    item = processed_results[i + j]
                    with cols[j]:
                        st.markdown(f"**Sample {item['idx']}:** <span class='green-code'>{item['name']}</span>", unsafe_allow_html=True)
                        st.image(item['res_plotted_rgb'], use_container_width=True)

    # --- TAB 2: SINGLE SAMPLE INSPECTION ---
    with tab2:
        st.subheader("🔍 Detailed Sample Inspection")
        st.info("💡 **Accuracy Tip:** If a subtle defect isn't highlighted, glide the **Confidence Threshold** slider down in the left sidebar to around **0.15–0.25** to catch lighter fabric flaws!")

        sample_names = [f"Sample {item['idx']}: {item['name']}" for item in processed_results]
        selected_sample_idx = st.selectbox("Select Sample to View:", range(len(sample_names)), format_func=lambda x: sample_names[x])

        selected_item = processed_results[selected_sample_idx]

        # Extract image safely and convert to PIL for Grad-CAM
        if 'img_rgb' in selected_item:
            orig_pil = Image.fromarray(selected_item['img_rgb'])
        elif 'orig_pil' in selected_item:
            orig_pil = selected_item['orig_pil']
        else:
            orig_pil = Image.fromarray(selected_item['res_plotted_rgb'])

        cam_overlay, _ = generate_cam_heatmap(model, orig_pil)

        # Display Bounding Box and Grad-CAM side-by-side
        st.subheader("🔍 Defect Analysis & Explainable AI (XAI)")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.image(selected_item['res_plotted_rgb'], caption="YOLOv8 Defect Detection (Bounding Boxes)", use_container_width=True)
            
        with col_b:
            st.image(cam_overlay, caption="Grad-CAM / Feature Attention Map (Hotspot Detection)", use_container_width=True)

        st.info("💡 **Explainability Note:** Red/yellow regions highlight where the model focused feature attention to flag defects.")

        st.markdown("---")

        scol1, scol2 = st.columns(2)
        with scol1:
            st.write("**Original Input Image**")
            st.image(selected_item['img_rgb'], use_container_width=True)

        with scol2:
            st.write(f"**Detection Overlay** (`{selected_model_name}`)")
            st.image(selected_item['res_plotted_rgb'], use_container_width=True)

            res_pil = Image.fromarray(selected_item['res_plotted_rgb'])
            buf = io.BytesIO()
            res_pil.save(buf, format="JPEG")
            st.download_button(
                label=f"📥 Download Annotated Sample ({selected_item['name']})",
                data=buf.getvalue(),
                file_name=f"inspected_{selected_item['name']}",
                mime="image/jpeg",
                key=f"single_dl_{selected_sample_idx}"
            )

    # --- TAB 3: AUDIT TRAIL, CSV & PDF ---
    with tab3:
        st.subheader("📋 Batch Quality Inspection Report")
        df_audit = pd.DataFrame([{
            "Sample #": item['idx'],
            "Filename": item['name'],
            "Quality Grade": item['grade'],
            "Defects Found": item['defects'],
            "Confidence Scores": item['confidences'],
            "Resolution": item['resolution'],
            "AI Latency (ms)": f"{item['latency']:.1f}"
        } for item in processed_results])

        st.dataframe(df_audit, use_container_width=True)

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            csv_bytes = df_audit.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Quality Audit Trail (CSV)",
                data=csv_bytes,
                file_name="quality_audit_trail.csv",
                mime="text/csv",
                key="audit_csv_download",
                use_container_width=True
            )

        with dcol2:
            summary_counts = {
                "total": len(uploaded_files),
                "grade_a": grade_a_count,
                "grade_b": grade_b_count,
                "grade_c": grade_c_count,
                "total_defects": total_defects
            }
            pdf_bytes = generate_pdf_report(processed_results, summary_counts, selected_model_name)
            st.download_button(
                label="📄 Export Quality Audit Report (PDF)",
                data=pdf_bytes,
                file_name="quality_inspection_report.pdf",
                mime="application/pdf",
                key="audit_pdf_download",
                use_container_width=True
            )

    # --- TAB 4: RESEARCH BENCHMARKS ---
    with tab4:
        st.subheader("🏆 Experimental Model Performance & Research Benchmarks")
        st.caption("Comparative evaluation across baseline deep learning architectures vs. the proposed CBAM + Soft Retinex enhanced pipeline.")

        benchmark_data = {
            "Model Pipeline": [
                "YOLOv8n + CBAM (Soft Retinex) [Proposed]",
                "YOLOv8n + CBAM (Full Retinex)",
                "YOLOv8n + CBAM (Ablation)",
                "YOLOv8n (Baseline)",
                "YOLO11n (Baseline)"
            ],
            "Recall (Sensitivity)": [0.4233, 0.3254, 0.3557, 0.3693, 0.3172],
            "mAP@50": [0.4461, 0.3422, 0.3453, 0.4666, 0.4097],
            "Inference Speed (FPS)": [115, 110, 122, 140, 135],
            "Params (M)": [3.15, 3.15, 3.15, 3.01, 2.58]
        }
        df_bench = pd.DataFrame(benchmark_data)
        st.dataframe(df_bench, use_container_width=True)

        models_labels = ['YOLOv8n\n(Baseline)', 'YOLO11n\n(Baseline)', 'YOLOv8n + CBAM\n(Ablation)', 'YOLOv8n + CBAM\n(Proposed)']
        recall_vals = [0.3693, 0.3172, 0.3557, 0.4233]
        map_vals = [0.4666, 0.4097, 0.3453, 0.4461]

        x = np.arange(len(models_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        rects1 = ax.bar(x - width/2, recall_vals, width, label='Recall (Defect Sensitivity)', color='#00d2ff')
        rects2 = ax.bar(x + width/2, map_vals, width, label='mAP@50 Score', color='#ffaa00')

        ax.set_ylabel('Performance Score', color='#ffffff')
        ax.set_title('Architecture Comparative Evaluation', color='#ffffff', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_labels, fontsize=8, color='#ffffff')
        ax.tick_params(colors='#ffffff')
        ax.legend(facecolor='#1b263b', edgecolor='#00d2ff', labelcolor='#ffffff')
        ax.set_ylim(0, 0.55)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        for rect in rects1 + rects2:
            h_val = rect.get_height()
            ax.annotate(f'{h_val:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h_val),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7, color='#ffffff')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        > **Key Research Insights:**
        > * **Recall Boost:** Integrating the **CBAM Attention Module** with **Soft Retinex** yields the highest recall score (**0.4233**), successfully capturing subtle fabric flaws (slubs, thin threads, light stains) that baseline YOLOv8n missed.
        > * **Real-Time Edge Deployment:** The proposed model retains real-time performance (>110 FPS on standard GPU setups), making it suitable for inline loom inspection systems.
        """)