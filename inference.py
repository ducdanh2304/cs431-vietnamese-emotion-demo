from pathlib import Path
import joblib
import re
import unicodedata
import numpy as np

LABEL_VI = {
    'Anger':'Tức giận', 'Disgust':'Ghê tởm/Khó chịu',
    'Enjoyment':'Vui vẻ', 'Fear':'Sợ hãi',
    'Other':'Khác/Không thuộc 6 cảm xúc', 'Sadness':'Buồn bã',
    'Surprise':'Ngạc nhiên',
}

def normalize_text(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(s))).strip()

class EmotionPredictor:
    def __init__(self, model_key='linear_svm', root=None, device='cpu'):
        self.root = Path(root) if root else Path(__file__).resolve().parent
        if model_key == 'balanced':
            import json
            model_key=json.loads((self.root/'selection.json').read_text(encoding='utf-8'))['balanced']
        self.model_key=model_key
        self.device=device
        if model_key in ('linear_svm','logreg','linear_svm_tuned','logreg_tuned'):
            obj=joblib.load(self.root/'models'/f'{model_key}.joblib')
            self.vectorizer=obj['vectorizer']
            self.clf=obj['classifier']
            self.labels=obj['labels']
        elif model_key=='visobert':
            import torch
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            self.torch=torch
            path=self.root/'models'/'visobert'
            self.tokenizer=AutoTokenizer.from_pretrained(path)
            self.model=AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
            self.labels=[self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        else:
            raise ValueError('model_key must be: logreg, linear_svm, visobert')

    def predict(self,text):
        s=normalize_text(text)
        if not s:
            raise ValueError('Vui lòng nhập câu có nội dung.')
        if self.model_key in ('logreg','linear_svm','logreg_tuned','linear_svm_tuned'):
            x=self.vectorizer.transform([s])
            idx=int(self.clf.predict(x)[0])
            score=(float(np.max(self.clf.predict_proba(x)[0]))
                   if hasattr(self.clf,'predict_proba') else None)
        else:
            with self.torch.inference_mode():
                enc=self.tokenizer(s,return_tensors='pt',truncation=True,max_length=128)
                enc={k:v.to(self.device) for k,v in enc.items()}
                probs=self.model(**enc).logits.softmax(-1)[0]
                idx=int(probs.argmax().item())
                score=float(probs[idx].item())
        label=self.labels[idx]
        return {'label':label,'label_vi':LABEL_VI.get(label,label),
                'confidence':score,'model':self.model_key}

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='balanced',choices=['balanced','linear_svm','logreg','visobert','linear_svm_tuned','logreg_tuned'])
    p.add_argument('--text',default='Hôm nay thật tuyệt vời, mình rất vui!')
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    predictor=EmotionPredictor(model_key=args.model,device=args.device)
    print(json.dumps(predictor.predict(args.text),ensure_ascii=False,indent=2))
