"""CS431 - Vietnamese Emotion Classifier (UIT-VSMEC) - Streamlit demo.

Chạy: streamlit run app.py
Dùng nguyên EmotionPredictor trong inference.py, không thay đổi preprocessing.
"""
import csv
import json
import time
from pathlib import Path

import streamlit as st

from inference import EmotionPredictor

ROOT = Path(__file__).resolve().parent
MAX_CHARS = 2000

st.set_page_config(page_title="Vietnamese Emotion Classifier", page_icon="🎭", layout="wide")

# ---------------------------------------------------------------- metadata
EMOTION = {
    "Enjoyment": ("😄", "#2e9e5b"),
    "Sadness": ("😢", "#3b82c4"),
    "Anger": ("😡", "#d64545"),
    "Fear": ("😨", "#8a5cc2"),
    "Surprise": ("😲", "#e0a100"),
    "Disgust": ("🤢", "#7a8b2e"),
    "Other": ("😐", "#6b7280"),
}
MODEL_INFO = {
    "linear_svm_tuned": ("Linear SVM (TF-IDF, tuned)", "ML nhanh nhất trên CPU"),
    "linear_svm": ("Linear SVM (TF-IDF)", "ML baseline"),
    "logreg_tuned": ("Logistic Regression (TF-IDF, tuned)", "ML, có điểm tin cậy"),
    "logreg": ("Logistic Regression (TF-IDF)", "ML baseline, có điểm tin cậy"),
    "visobert": ("ViSoBERT (fine-tuned)", "DL, chính xác nhất"),
}
EXAMPLES = {
    "😄 Vui": "Hôm nay mình vui quá!",
    "😢 Buồn": "Mình buồn vì chuyện vừa xảy ra.",
    "😡 Giận": "Thật bực mình!",
    "😨 Sợ": "Tôi sợ quá!",
    "😍 Emoji": "Hạnh phúc quá 😍",
    "🗣️ Từ lóng": "hnay đi học vui vãi chưởng =))",
    "📜 Câu dài": "Hôm nay trời mưa, mình đi học muộn, bị thầy nhắc, rồi lại làm mất ví. " * 12,
}
DEFAULT_TEXT = EXAMPLES["😄 Vui"]


def label_name(key: str) -> str:
    return MODEL_INFO.get(key, (key, ""))[0]


# ---------------------------------------------------------------- data
@st.cache_data
def load_benchmark() -> dict:
    """Số liệu test + latency CPU đã đo trong notebook (evidence/final_comparison.csv)."""
    path = ROOT / "evidence" / "final_comparison.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as f:
        return {r["model"]: r for r in csv.DictReader(f)}


@st.cache_data
def load_selection() -> dict:
    p = ROOT / "selection.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def available_models() -> list:
    keys = sorted(p.stem for p in (ROOT / "models").glob("*.joblib"))
    if (ROOT / "models" / "visobert" / "config.json").exists():
        keys.append("visobert")
    order = list(MODEL_INFO)
    return sorted(keys, key=lambda k: order.index(k) if k in order else 99)


@st.cache_resource(show_spinner=False)
def get_predictor(model_key: str, device: str) -> EmotionPredictor:
    return EmotionPredictor(model_key=model_key, root=ROOT, device=device)


def run_model(model_key: str, text: str, device: str) -> dict:
    """Load (cache) + predict. Trả về kết quả kèm latency của riêng lần predict."""
    predictor = get_predictor(model_key, device if model_key == "visobert" else "cpu")
    t0 = time.perf_counter()
    result = predictor.predict(text)
    result["ms"] = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------- UI helpers
st.markdown(
    """
    <style>
    .result-card{border-radius:16px;padding:22px 26px;color:#fff;margin:6px 0 10px}
    .result-card .emo{font-size:54px;line-height:1.1}
    .result-card .vi{font-size:30px;font-weight:700}
    .result-card .en{opacity:.85;font-size:15px}
    .small-card{border-radius:14px;padding:14px 16px;color:#fff;min-height:132px}
    .small-card .vi{font-size:20px;font-weight:700}
    .small-card .meta{font-size:13px;opacity:.9}
    </style>
    """,
    unsafe_allow_html=True,
)


def conf_text(result: dict) -> str:
    c = result.get("confidence")
    return "không hỗ trợ (SVM không có xác suất)" if c is None else f"{float(c):.1%} (chưa hiệu chuẩn)"


def show_result_card(result: dict) -> None:
    emoji, color = EMOTION.get(result["label"], ("🎭", "#444"))
    st.markdown(
        f"""<div class="result-card" style="background:{color}">
        <div class="emo">{emoji}</div>
        <div class="vi">{result['label_vi']}</div>
        <div class="en">{result['label']}</div></div>""",
        unsafe_allow_html=True,
    )
    c = result.get("confidence")
    if c is not None:
        st.progress(min(max(float(c), 0.0), 1.0), text=f"Điểm softmax/xác suất: {float(c):.1%} (chưa hiệu chuẩn)")
    else:
        st.info("Mô hình này không trả điểm tin cậy.")


def validate(text: str):
    if not text or not text.strip():
        return "Vui lòng nhập văn bản."
    if len(text) > MAX_CHARS:
        return f"Văn bản quá dài ({len(text)} ký tự). Giới hạn của demo là {MAX_CHARS} ký tự."
    return None


# ---------------------------------------------------------------- sidebar
models = available_models()
if not models:
    st.error("Không tìm thấy checkpoint trong thư mục models/.")
    st.stop()

bench = load_benchmark()
selection = load_selection()

with st.sidebar:
    st.header("⚙️ Cấu hình")
    preferred = selection.get("balanced")
    model_key = st.selectbox(
        "Mô hình",
        models,
        index=models.index(preferred) if preferred in models else 0,
        format_func=label_name,
    )
    st.caption(MODEL_INFO.get(model_key, ("", ""))[1])

    device = "cpu"
    if "visobert" in models:
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        device = st.radio("Thiết bị cho ViSoBERT", ["cpu", "cuda"] if has_cuda else ["cpu"], horizontal=True)

    st.divider()
    st.markdown(
        "**Dữ liệu:** UIT-VSMEC, 7 nhãn  \n"
        "**ML:** TF-IDF + Linear SVM / Logistic Regression  \n"
        "**DL:** ViSoBERT fine-tuned, max_length 128"
    )

# ---------------------------------------------------------------- header
st.title("🎭 Vietnamese Emotion Classifier")
st.caption("Nhập một câu tiếng Việt, mô hình dự đoán 1 trong 7 cảm xúc của UIT-VSMEC.")

tab_predict, tab_compare, tab_bench = st.tabs(["🔍 Dự đoán", "⚖️ So sánh ML vs DL", "📊 Kết quả thực nghiệm"])

if "text" not in st.session_state:
    st.session_state["text"] = DEFAULT_TEXT


def set_example(value: str) -> None:
    st.session_state["text"] = value


def text_input_block(key_prefix: str) -> str:
    st.write("**Câu ví dụ**")
    cols = st.columns(len(EXAMPLES))
    for i, (name, value) in enumerate(EXAMPLES.items()):
        cols[i].button(name, key=f"{key_prefix}_ex_{i}", on_click=set_example, args=(value,))
    text = st.text_area("Văn bản tiếng Việt", key="text", height=130, placeholder="Ví dụ: Hôm nay mình vui quá!")
    st.caption(f"{len(text)} / {MAX_CHARS} ký tự")
    return text


# ---------------------------------------------------------------- tab 1
with tab_predict:
    text = text_input_block("p")
    if st.button("Dự đoán", type="primary", key="btn_predict"):
        error = validate(text)
        if error:
            st.warning(error)
        else:
            try:
                with st.spinner("Đang tải mô hình (chỉ lần đầu) và dự đoán..."):
                    result = run_model(model_key, text, device)
            except Exception as exc:  # hiển thị lỗi thay vì crash
                st.error(f"Không thể suy luận: {exc}")
            else:
                left, right = st.columns([1, 1])
                with left:
                    show_result_card(result)
                with right:
                    st.metric("Mô hình", label_name(result["model"]))
                    st.metric("Thời gian dự đoán (lần chạy này)", f"{result['ms']:.1f} ms")
                    if result["model"] == "visobert" and len(text) > 400:
                        st.caption("Câu dài sẽ được cắt ở 128 token theo cấu hình huấn luyện.")
                st.caption(
                    "Độ trễ trên là đo một lần trong demo, không phải benchmark. "
                    "Benchmark chính thức nằm ở tab Kết quả thực nghiệm."
                )

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.write("Chạy cùng một câu qua mô hình ML nhanh và ViSoBERT để thấy sự đánh đổi độ chính xác và tốc độ.")
    compare_keys = [k for k in ("linear_svm_tuned", "logreg_tuned", "visobert") if k in models]
    chosen = st.multiselect("Mô hình so sánh", models, default=compare_keys, format_func=label_name)
    ctext = st.text_area("Văn bản", value=st.session_state["text"], height=100, key="cmp_text")
    if st.button("So sánh", type="primary", key="btn_compare"):
        error = validate(ctext)
        if error:
            st.warning(error)
        elif not chosen:
            st.warning("Chọn ít nhất một mô hình.")
        else:
            cols = st.columns(len(chosen))
            for col, key in zip(cols, chosen):
                with col:
                    try:
                        with st.spinner(f"{label_name(key)}..."):
                            r = run_model(key, ctext, device)
                        emoji, color = EMOTION.get(r["label"], ("🎭", "#444"))
                        conf = r["confidence"]
                        st.markdown(
                            f"""<div class="small-card" style="background:{color}">
                            <div class="meta">{label_name(key)}</div>
                            <div class="vi">{emoji} {r['label_vi']}</div>
                            <div class="meta">Điểm: {'—' if conf is None else f'{conf:.1%}'}<br>
                            Thời gian: {r['ms']:.1f} ms</div></div>""",
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.error(f"{label_name(key)}: {exc}")
            st.caption(
                "Lần chạy đầu của mỗi mô hình gồm cả warm-up nên có thể chậm hơn. "
                "Số liệu latency so sánh đúng nằm ở tab Kết quả thực nghiệm."
            )

# ---------------------------------------------------------------- tab 3
with tab_bench:
    st.write("Số liệu từ notebook của team, tập test UIT-VSMEC. Latency đo trên CPU, batch=1.")
    if bench:
        import pandas as pd

        rows = []
        for key in ("logreg", "linear_svm", "logreg_tuned", "linear_svm_tuned", "visobert"):
            r = bench.get(key)
            if not r:
                continue
            rows.append(
                {
                    "Mô hình": label_name(key),
                    "Accuracy": float(r["test_accuracy"]),
                    "Macro-F1": float(r["test_macro_f1"]),
                    "Latency p50 (ms)": float(r["cpu_p50_ms"]),
                    "Latency p95 (ms)": float(r["cpu_p95_ms"]),
                    "Kích thước (MB)": float(r["artifact_size_mb"]),
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.info(
            "ViSoBERT cao hơn ML tốt nhất khoảng 9 điểm Accuracy (0.693 so với 0.600) "
            "và 7 điểm Macro-F1 (0.663 so với 0.590), nhưng chậm hơn khoảng 33 lần trên CPU "
            "(p50 55.5 ms so với 1.7 ms) và nặng hơn khoảng 108 lần (374 MB so với 3.5 MB)."
        )
    else:
        st.warning("Không tìm thấy evidence/final_comparison.csv.")

    imgs = [
        ("accuracy_latency_tradeoff.png", "Accuracy và latency"),
        ("test_accuracy_macro_f1.png", "Accuracy và Macro-F1"),
        ("confusion_visobert.png", "Confusion matrix ViSoBERT"),
    ]
    cols = st.columns(len(imgs))
    for col, (fname, cap) in zip(cols, imgs):
        p = ROOT / "evidence" / fname
        if p.exists():
            col.image(str(p), caption=cap)
