FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create uploads directory
RUN mkdir -p uploads

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "resume:server"] 