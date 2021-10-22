FROM python:3.7-slim-buster

USER root
RUN mkdir /src

COPY app /src/app
COPY model /src/model
COPY requirements.txt /src/

WORKDIR /src

RUN python --version
RUN pip install -r requirements.txt

CMD PYTHONPATH="." python app/app_main.py