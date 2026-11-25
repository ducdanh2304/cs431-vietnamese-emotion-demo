# CS431 Emotion Classifier — Demo Package

## Cài đặt

python -m pip install -r requirements_app.txt

## Kiểm tra

python demo_smoke.py

## Chạy giao diện

streamlit run demo_app.py

## Ghi chú

- FULL chứa ViSoBERT và các model ML.
- ML_LITE chỉ chứa các model ML.
- Không cần train lại.
- Input: Vietnamese text.
- Output: emotion label và score nếu khả dụng.
- Inference latency notebook nằm trong gói báo cáo nội dung.
