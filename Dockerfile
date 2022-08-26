FROM python:3.8

RUN apt-get update \
    && apt-get install curl -y \
    && curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /code

COPY Makefile pyproject.toml poetry.lock /code/
RUN make setup

COPY app/ /code/app/

CMD ["make", "run-prod"]
