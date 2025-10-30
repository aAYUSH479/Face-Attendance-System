FROM python:3.11-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y cmake g++ make libboost-all-dev

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
