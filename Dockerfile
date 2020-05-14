FROM rbarile17/py_cb_recsys

COPY src ./src
COPY test ./test

ENV PYTHONPATH "${PYTHONPATH}:./src"
