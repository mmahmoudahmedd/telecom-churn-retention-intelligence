FROM python:3.13-slim
WORKDIR /app
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt
COPY app.py retention.py /app/
COPY reports/ /app/reports/
ENV PORT=8501
EXPOSE 8501
CMD ["sh", "-c", "python -m streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true --browser.gatherUsageStats false"]
