FROM python:3.9-bullseye AS base
WORKDIR /code
RUN pip install --upgrade pip
COPY ./requirements ./requirements
RUN pip install --no-cache-dir -r ./requirements/base.txt


FROM base AS prod
RUN pip install --no-cache-dir -r ./requirements/prod.txt
COPY ./src .

FROM prod AS test
RUN pip install --no-cache-dir -r ./requirements/test.txt
COPY ./tests .
COPY .space.kts .

FROM test as dev
RUN pip install --no-cache-dir -r ./requirements/dev.txt
