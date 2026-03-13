FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir . uvicorn[standard] fastapi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.web:app", "--host", "0.0.0.0", "--port", "8000"]
