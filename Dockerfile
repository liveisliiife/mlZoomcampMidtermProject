FROM python:3.10.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["predict.py","model_midterm.bin","./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve","--listen=0.0.0.0:9696","predict:app" ]