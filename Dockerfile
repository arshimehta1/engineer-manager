FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV ENABLE_WEB_INTERFACE=true

COPY pyproject.toml README.md /app/
COPY server /app/server
COPY models.py __init__.py /app/

RUN pip install --upgrade pip && \
    pip install .

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
