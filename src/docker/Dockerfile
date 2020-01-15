FROM python:3.7

RUN apt-get update && apt-get install -y make python-pip

RUN make .
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app
EXPOSE 5050

CMD python ./main.py
