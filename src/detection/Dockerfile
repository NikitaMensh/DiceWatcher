FROM ultralytics/ultralytics:latest-cpu

WORKDIR /api

COPY ./src/detection/requirements.txt /api/requirements.txt

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY ./src/detection /api
COPY ./models /api/models

EXPOSE 8008

CMD ["uvicorn", "detection:app", "--host", "0.0.0.0", "--port", "8000"]
