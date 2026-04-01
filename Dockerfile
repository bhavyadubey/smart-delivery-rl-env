FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir numpy pydantic

CMD ["python", "baseline/run_baseline.py"]
