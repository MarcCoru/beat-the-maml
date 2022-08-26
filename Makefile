UID := $(shell id -u)
GID := $(shell id -g)
FLASK_APP := app
FLASK_RUN_HOST := 0.0.0.0
FLASK_RUN_PORT := 5000

generate-selfsigned-cert:
	cd cert && OWNER="${UID}.${GID}" docker-compose up --remove-orphans

setup: # local & inside Docker
	poetry install

run-dev: # local
	cd app; FLASK_APP=${FLASK_APP} FLASK_RUN_HOST=${FLASK_RUN_HOST} FLASK_RUN_PORT=${FLASK_RUN_PORT} FLASK_ENV=development poetry run flask run

run-prod: # local & inside Docker
	cd app; poetry run gunicorn --bind ${FLASK_RUN_HOST}:${FLASK_RUN_PORT} wsgi:app

run: # main entry for on server, can be run on desktop also
	docker-compose build --pull
	docker-compose up --remove-orphans --force-recreate -d
