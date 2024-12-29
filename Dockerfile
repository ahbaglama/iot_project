FROM python:3.9

WORKDIR /app

COPY ./app /app/app
COPY ./model /app/model
COPY ./templates /app/templates
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
