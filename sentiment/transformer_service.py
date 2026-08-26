import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_tokenizer = None
_model = None


def get_model():

    global _tokenizer
    global _model

    if _model is None:

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        _model.eval()

    return _tokenizer, _model


def predict_sentiment(text: str):

    tokenizer, model = get_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    predicted_id = int(
        torch.argmax(probabilities)
    )

    label = model.config.id2label[
        predicted_id
    ].lower()

    confidence = float(
        probabilities[predicted_id]
    )

    return {
        "sentiment": label,
        "confidence": round(
            confidence,
            4,
        ),
        "model": MODEL_NAME,
    }