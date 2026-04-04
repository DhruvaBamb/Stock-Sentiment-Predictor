FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
