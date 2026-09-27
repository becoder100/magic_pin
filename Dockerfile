FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Hosting platforms (Railway, HF Spaces) inject PORT; default 8080 for local runs.
ENV PORT=8080
EXPOSE 8080

# Exactly ONE worker: all state (contexts, conversations) lives in memory,
# so a second worker would have its own empty copy.
CMD ["sh", "-c", "uvicorn bot:app --host 0.0.0.0 --port ${PORT} --workers 1"]
