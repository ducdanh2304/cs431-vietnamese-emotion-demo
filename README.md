# CS431 – Vietnamese Emotion Classifier (Demo)

Demo Streamlit phân loại cảm xúc câu tiếng Việt thành 7 nhãn của UIT-VSMEC
(Anger, Disgust, Enjoyment, Fear, Other, Sadness, Surprise).
Có thể chuyển giữa mô hình ML (TF-IDF + Linear SVM / Logistic Regression) và DL (ViSoBERT) để so sánh độ chính xác và độ trễ.

## Tính năng

- Nhập văn bản tiếng Việt, dự đoán nhãn (tiếng Anh và tiếng Việt), điểm tin cậy nếu mô hình hỗ trợ, thời gian dự đoán.
- 7 câu ví dụ có sẵn: vui, buồn, giận, sợ, emoji, từ lóng, câu dài.
- Tab **So sánh ML vs DL**: chạy một câu qua nhiều mô hình cùng lúc.
- Tab **Kết quả thực nghiệm**: bảng Accuracy, Macro-F1, latency và biểu đồ từ notebook của team.
- Xử lý input rỗng và input quá 2000 ký tự. Model chỉ được tải một lần (cache).

## Cấu trúc

```
app.py                  Giao diện Streamlit (file chạy chính)
inference.py            EmotionPredictor, do Thúy xuất từ notebook (không sửa)
demo_smoke.py           Kiểm thử nhanh tải model và dự đoán
models/*.joblib         4 model ML (vectorizer + classifier)
models/visobert/        ViSoBERT fine-tuned (xem mục "Tải ViSoBERT")
evidence/               Bảng kết quả và biểu đồ dùng cho tab Kết quả thực nghiệm
labels.json, selection.json, config.json
requirements_app.txt    Thư viện cần cài
```

## Yêu cầu

- Python 3.10 trở lên.
- Phải dùng đúng phiên bản trong `requirements_app.txt`. Đặc biệt `transformers==4.57.3`: bản 5.x làm lỗi tokenizer của ViSoBERT
  (`'dict' object is not an instance of 'Sequence'`).

## Cài đặt

```powershell
git clone <URL-repo>
cd <ten-repo>
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements_app.txt
```

Linux/macOS:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements_app.txt
```

## Tải ViSoBERT (bắt buộc nếu muốn chạy mô hình DL)

File `models/visobert/model.safetensors` nặng khoảng 381 MB, vượt giới hạn 100 MB của GitHub nên không nằm trong repo.
Lấy từ gói `CS431_DEMO_FULL.zip` của team, rồi chép các file sau vào `models/visobert/`:

```
model.safetensors
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
sentencepiece.bpe.model
```

Thiếu thư mục này, app vẫn chạy với 4 model ML (đúng với bản ML_LITE). ViSoBERT chỉ hiện trong danh sách khi có `models/visobert/config.json`.

## Chạy

Kiểm tra nhanh (tùy chọn):

```powershell
.\.venv\Scripts\python demo_smoke.py
```

Chạy giao diện:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Mở `http://localhost:8501`. Tắt bằng `Ctrl+C` trong terminal.

Nên chạy bằng `python -m streamlit` của venv như trên. Gõ `streamlit run app.py` trực tiếp có thể dùng Python toàn cục với phiên bản thư viện sai.

## Cách dùng

1. Chọn mô hình ở thanh bên trái (mặc định là ViSoBERT nếu có).
2. Bấm một câu ví dụ hoặc tự nhập câu tiếng Việt.
3. Bấm **Dự đoán**.
4. Qua tab **So sánh ML vs DL** để so cùng một câu trên nhiều mô hình.

## Kết quả thực nghiệm (tập test UIT-VSMEC)

| Mô hình | Accuracy | Macro-F1 | CPU p50 (ms) | Kích thước (MB) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.582 | 0.578 | 2.46 | 3.6 |
| Linear SVM | 0.589 | 0.568 | 1.70 | 3.3 |
| Logistic Regression (tuned) | 0.582 | 0.578 | 2.56 | 3.6 |
| Linear SVM (tuned) | 0.600 | 0.590 | 1.70 | 3.5 |
| ViSoBERT | 0.693 | 0.663 | 55.5 | 374.4 |

ViSoBERT cao hơn ML tốt nhất khoảng 9 điểm Accuracy, nhưng chậm hơn khoảng 33 lần trên CPU và nặng hơn khoảng 108 lần.
Latency lấy từ benchmark của notebook (CPU, batch=1), khác với thời gian hiển thị trong demo.

## Lưu ý khi demo

- Linear SVM không có xác suất nên ô điểm hiển thị "—". Không tự chuyển margin thành xác suất.
- Điểm của Logistic Regression và ViSoBERT là điểm softmax chưa hiệu chuẩn.
- Lần dự đoán đầu của mỗi mô hình chậm hơn vì phải tải model và warm-up.
- Một số câu tự tạo bị dự đoán sai (ví dụ "Thật bực mình!" được ViSoBERT xếp Disgust thay vì Anger). Đây là giới hạn của mô hình.
- `scikit-learn` trong `requirements_app.txt` được ghim 1.6.1 để khớp phiên bản lúc train các file `.joblib`.

## Nguồn

Model và `inference.py` do Thúy huấn luyện và xuất từ notebook (dataset `tridm/UIT-VSMEC`, backbone `uitnlp/visobert`).
`app.py` do Danh viết.
