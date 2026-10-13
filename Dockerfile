FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt

COPY dashboard/ ./dashboard/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

CMD streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
