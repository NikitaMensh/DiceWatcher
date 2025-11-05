FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./src/app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/app/app.py app.py
COPY ./src/mechanics/mechanics.py mechanics.py
ENV API_URL=http://api:8008
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]