FROM python:3.7-buster
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY train.py /app/train.py
ENTRYPOINT ["python", "train.py"]