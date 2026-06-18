FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p vectorstore data

EXPOSE 7860 8501

CMD ["bash", "-c", "python main.py & streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]