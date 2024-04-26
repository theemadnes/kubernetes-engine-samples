#!/bin/sh

echo "Checking if HTTP or gRPC mode"
if [ -n "${GRPC_ENABLED}" ] && [ "${GRPC_ENABLED}" == "True" ]
then
    echo "gRPC mode enabled"
    python app.py
else
    echo "Flask/HTTP mode enabled"
    gunicorn --config gunicorn.conf.py app:app --log-config gunicorn_logging.conf --worker-tmp-dir /tmp
fi
