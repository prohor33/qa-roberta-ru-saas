FROM qts8n/cuda-python:runtime

USER root
RUN mkdir /src

COPY app /src/app
COPY model /src/model
COPY requirements.txt /src/

WORKDIR /src

RUN python --version
RUN pip install -r requirements.txt

CMD gunicorn --config app/conf/gunicorn.py "app.app_main:create_app()"