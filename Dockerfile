FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py check_threshold.py ./

RUN echo "Fetching model for run ${RUN_ID}" > /tmp/fetch.log

CMD ["python", "-c", "print('Container ready for run ' + '${RUN_ID}')"]