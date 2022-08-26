#!/bin/sh

KEY_FILE=/root/certificate.key
CRT_FILE=/root/certificate.crt

openssl req \
    -newkey rsa:2048 \
    -x509 \
    -nodes \
    -keyout ${KEY_FILE} \
    -new \
    -out ${CRT_FILE} \
    -subj '/CN=localhost' \
    -sha256 \
    -days 365

if [ ! -z "${OWNER}" ]; then
  chown ${OWNER} ${KEY_FILE} ${CRT_FILE};
fi;
