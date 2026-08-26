FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1200 \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=1200 \
    -r requirements.txt

COPY api ./api
COPY recommendation ./recommendation
COPY sentiment ./sentiment
COPY llm ./llm
COPY mlflow_tracking ./mlflow_tracking
COPY data/processed ./data/processed

COPY mlflow.db .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]