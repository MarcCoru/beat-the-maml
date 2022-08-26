# Flask Hello world project ready with docker

Tiny demo project that run a Flask hello app :

- locally with Flask command (dev)
- locally with Gunicorn (test)
- locally or on server with Gunicorn inside Docker (prod)

# Run the app

## locally

Run the following commands :

```bash
make setup # only once
make generate-selfsigned-cert # only once

make run-dev # dev mode (Flask)
make run-prod # test mode (Gunicorn)
make run # prod mode (Gunicorn inside Docker)
```

## on server

Run the following command :

```bash
make run # prod mode (Gunicorn inside Docker)
```

Stop the service :

```bash
docker-compose down
```

## Check result

- dev / test : browse to http://localhost:5000
- prod : browse to https://localhost (you'll have to accept self signed certificate)
