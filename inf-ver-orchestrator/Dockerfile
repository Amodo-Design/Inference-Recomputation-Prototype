FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

CMD ["python3", "-m", "app.main"]
