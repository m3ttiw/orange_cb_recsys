FROM rbarile17/py_cb_recsys

COPY src .
COPY test .

RUN export PYTHONPATH=$PYTHONPATH:$(pwd)
