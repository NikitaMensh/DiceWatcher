FROM python:3.11-slim
WORKDIR /app
COPY ./src/app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/app/app.py app.py
ENV API_URL=http://api:8008
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]