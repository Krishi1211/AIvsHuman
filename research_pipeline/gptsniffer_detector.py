from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


class CodeBERTLogRegDetector:
    """A lightweight GPTSniffer-style detector based on CodeBERT embeddings."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 192,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()
        self.classifier = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        )

    def _mean_pool(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_sum = torch.sum(token_embeddings * mask, dim=1)
        mask_sum = torch.clamp(mask.sum(dim=1), min=1e-9)
        return masked_sum / mask_sum

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.encoder(**encoded)
                pooled = self._mean_pool(output, encoded["attention_mask"])
                embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        vectors = self.embed_texts(texts)
        self.classifier.fit(vectors, labels)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.embed_texts(texts)
        return self.classifier.predict_proba(vectors)[:, 1]

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, output_path / "logreg.joblib")
        (output_path / "detector_config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "batch_size": self.batch_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> "CodeBERTLogRegDetector":
        output_path = Path(output_dir)
        config = json.loads((output_path / "detector_config.json").read_text(encoding="utf-8"))
        detector = cls(
            model_name=config["model_name"],
            max_length=config["max_length"],
            batch_size=config["batch_size"],
        )
        detector.classifier = joblib.load(output_path / "logreg.joblib")
        return detector
