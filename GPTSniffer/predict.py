import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["ChatGPT", "Human"]  # matches your target_names

def predict(text: str, model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.to(device)
    mdl.eval()

    enc = tok(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    return probs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./saved_model")
    ap.add_argument("--file", help="Path to a code file to classify")
    args = ap.parse_args()

    if not args.file:
        raise SystemExit("Usage: python predict.py --file example.py")

    with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    probs = predict(text, args.model_dir)
    for lab, p in zip(LABELS, probs):
        print(f"{lab}: {p:.4f}")
