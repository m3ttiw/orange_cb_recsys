FROM rbarile17/py_cb_recsys

COPY src .
COPY test .

ENV PYTHONPATH "${PYTHONPATH}:./src"
