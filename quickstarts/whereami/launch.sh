#!/bin/sh

echo "Checking if HTTP or gRPC mode"
if [ "$GRPC_ENABLED" == "True" ]
then
    echo "gRPC mode enabled"
    python app.py
else
    echo "Flask/HTTP mode enabled"
    echo "Attempting to get CPU count"
    CORE_COUNT=$(grep --count ^processor /proc/cpuinfo)

    if [ -z "$CORE_COUNT" ]
    then
        echo "\$CORE_COUNT is NULL - setting to default of 4"
        GUNICORN_THREADS=4
    else
        echo "\$CORE_COUNT is NOT NULL"
        GUNICORN_THREADS=$(( 2* $CORE_COUNT ))
    fi

    # handle host between IPv4 and IPv6
    if [ -z "$HOST" ]
    then
        echo "\$HOST variable found"
    else
        echo "\$HOST variable not found"
        HOST="0.0.0.0"
    fi

    echo "Setting Gunicorn port"
    if [ -z "$PORT" ]
    then
        echo "\$PORT is NULL - setting to default of 8080"
        gunicorn -w $GUNICORN_THREADS -b $HOST:8080 app:app
    else
        echo "\$CORE_COUNT is NOT NULL"
        gunicorn -w $GUNICORN_THREADS -b $HOST:${PORT} app:app
    fi
fi


