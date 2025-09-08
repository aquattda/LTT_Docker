FROM ubuntu

WORKDIR /src

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-sklearn

COPY iris_ml.py ./iris_ml.py 

CMD ["python3", "iris_ml.py"]