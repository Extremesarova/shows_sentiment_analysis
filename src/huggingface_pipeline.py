import gc
import math
from typing import Any, List

import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tokenizer:
    def __init__(
        self,
        model_name: str = "Tatyana/rubert-base-cased-sentiment-new",
        max_length=512,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
    ) -> List[str]:

        inputs = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(DEVICE)

        return inputs


class Model:
    def __init__(
        self,
        model_name: str = "Tatyana/rubert-base-cased-sentiment-new",
    ) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            DEVICE
        )

    def get_logits(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits

        return logits

    def logits_to_labels(self, logits) -> List[str]:
        predicted_class_ids = logits.argmax(axis=1).cpu().detach().numpy()

        pred_labels = [
            self.model.config.id2label[predicted_class_id]
            for predicted_class_id in predicted_class_ids
        ]

        return pred_labels


class InferencePipeline:
    def __init__(
        self,
        texts: List[str],
        class_labels: List[Any],
        model_name: str = "Tatyana/rubert-base-cased-sentiment-new",
        max_length=512,
        batch_size=168,
        correction_map: dict = {
            "POSITIVE": "positive",
            "NEUTRAL": "neutral",
            "NEGATIVE": "negative",
        },
    ) -> None:
        self.texts = texts
        self.class_labels = class_labels
        self.pred_labels = []

        self.max_length = max_length
        self.batch_size = batch_size
        self.correction_map = correction_map
        self.num_batches = math.ceil(len(texts) / batch_size)

        self.model_name = model_name

        torch.cuda.empty_cache()
        gc.collect()

        self.tokenizer = Tokenizer(model_name=model_name, max_length=max_length)
        self.model = Model(model_name=model_name)

    def generate_batches(self, X: List[Any], y: List[Any]):
        assert len(X) == len(y)

        for batch_start in range(0, len(X), self.batch_size):
            batch_end = batch_start + self.batch_size
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            yield X_batch, y_batch

    def correct_labels(self, labels: List[str]) -> List[str]:
        corrected_labels = [self.correction_map[label] for label in labels]

        return corrected_labels

    def texts_to_sentiments(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer.tokenize(texts)
        logits = self.model.get_logits(inputs)
        pred_labels = self.model.logits_to_labels(logits=logits)
        pred_labels = self.correct_labels(pred_labels)

        return pred_labels

    def batch_inference(self):
        print(
            f"Inferencing using {self.model_name} model with batch_size={self.batch_size}"
        )
        for texts_batch, _ in tqdm(
            self.generate_batches(self.texts, self.class_labels),
            total=self.num_batches,
            unit="batch",
        ):
            self.pred_labels.extend(
                self.texts_to_sentiments(
                    texts=list(texts_batch),
                )
            )

    @staticmethod
    def get_f1_score(y_true, y_pred, averaging="micro"):
        assert len(y_true) == len(y_pred), "Check labels"

        f1 = round(f1_score(y_true, y_pred, average=averaging), 3)
        return f1
