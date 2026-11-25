from pathlib import Path
from inference import EmotionPredictor

ROOT = Path(__file__).resolve().parent
models = sorted(p.stem for p in (ROOT / "models").glob("*.joblib"))
if not models and (ROOT / "models" / "visobert").exists():
    models = ["visobert"]

if not models:
    raise RuntimeError("Không tìm thấy model artifacts")

p = EmotionPredictor(model_key=models[0], root=ROOT, device="cpu")
result = p.predict("Hôm nay mình vui quá!")
assert result["label"] in {
    "Anger", "Disgust", "Enjoyment", "Fear",
    "Other", "Sadness", "Surprise"
}
print("DEMO_SMOKE_PASS", result)
