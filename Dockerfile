# syntax=docker/dockerfile:1

FROM python:3.9

WORKDIR /code

COPY requirements.txt /code/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /code/

CMD ["python3", "app.py"]
