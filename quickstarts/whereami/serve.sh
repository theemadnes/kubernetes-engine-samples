#!/bin/sh

echo "Checking if HTTP or gRPC mode"
if [ "$GRPC_ENABLED" == "True" ]
then
    echo "gRPC mode enabled"
    python app.py
else
    echo "Flask/HTTP mode enabled"
    gunicorn --log-config gunicorn_logging.conf --config gunicorn.conf.py app:app
fi
