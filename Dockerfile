# syntax=docker/dockerfile:1

FROM python:3.9
# FROM python:3.8-slim-buster
# ENV PYTHONUNBUFFERED=1
# RUN apt-get update
# RUN apt-get install -y libgl1-mesa-dev  libglib2.0-0

WORKDIR /code

COPY requirements.txt /code/
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install opencv-python-headless # x docker usare questo

COPY . /code/

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
CMD ["python3", "app.py"]
