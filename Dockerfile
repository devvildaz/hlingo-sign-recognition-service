FROM python:3.10

WORKDIR /hololingo-lsp

COPY ./requirements.txt ./requirements.txt

COPY ./.env ./.env

COPY ./model ./model

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY *.py ./

EXPOSE $PORT
CMD uvicorn main:app --host 0.0.0.0 --port 8000