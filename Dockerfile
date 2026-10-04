FROM python:3.11-slim
WORKDIR /app
# xgboost a besoin d'OpenMP (absent de l'image slim)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/model.json models/model.json
COPY reports/metrics.json reports/metrics.json
COPY config.json config.json
