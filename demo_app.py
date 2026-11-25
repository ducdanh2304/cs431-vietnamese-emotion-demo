from pathlib import Path
import json
import time
import streamlit as st
from inference import EmotionPredictor

ROOT = Path(__file__).resolve().parent

models = sorted(p.stem for p in (ROOT / "models").glob("*.joblib"))
if (ROOT / "models" / "visobert" / "config.json").exists():
    models.append("visobert")

selection_file = ROOT / "selection.json"
selection = json.loads(selection_file.read_text(encoding="utf-8")) if selection_file.exists() else {}
preferred = selection.get("balanced")
default = models.index(preferred) if preferred in models else 0

st.set_page_config(page_title="Vietnamese Emotion Classifier", page_icon="🎭")
st.title("🎭 Vietnamese Emotion Classifier")
st.write("Phân loại 7 cảm xúc tiếng Việt trên UIT-VSMEC")

@st.cache_resource(show_spinner=False)
def get_predictor(model_key, device):
    return EmotionPredictor(model_key=model_key, root=ROOT, device=device)

if not models:
    st.error("Không tìm thấy checkpoint trong models/")
    st.stop()

model_key = st.selectbox("Chọn mô hình", models, index=default)
device = "cpu"
if model_key == "visobert":
    device = st.selectbox("Thiết bị", ["cpu", "cuda"])

value = st.text_area("Nhập văn bản tiếng Việt", "Hôm nay mình vui quá!")
if st.button("Dự đoán", type="primary"):
    if not value.strip():
        st.warning("Vui lòng nhập văn bản.")
    else:
        try:
            predictor = get_predictor(model_key, device)
            t0 = time.perf_counter()
            result = predictor.predict(value)
            elapsed = (time.perf_counter() - t0) * 1000
            st.success(result.get("label_vi", result.get("label")))
            st.write("Label:", result.get("label"))
            st.write("Model:", result.get("model"))
            if result.get("confidence") is not None:
                st.write("Score chưa hiệu chuẩn:", round(float(result["confidence"]), 4))
            st.caption(
                f"Inference đơn lẻ: {elapsed:.2f} ms. "
                "Không thay thế benchmark latency đã lưu."
            )
        except Exception as exc:
            st.error(f"Không thể suy luận: {exc}")
