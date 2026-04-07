FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    numpy \
    "openenv-core[core]" \
    pydantic \
    streamlit \
    uvicorn

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "server.app", "--host", "0.0.0.0", "--port", "8000"]
