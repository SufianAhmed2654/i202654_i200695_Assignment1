FROM python:3.9-slim
ENV HTTP_TIMEOUT=1000000
WORKDIR /app
COPY app.py model.pkl requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV MODEL_PATH=model.pkl
CMD ["python", "app.py"]
