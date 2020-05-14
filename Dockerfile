FROM rbarile17/py_cb_recsys

COPY src ./src
COPY test ./test

ENV PYTHONPATH "${PYTHONPATH}:./src"

RUN chmod a+x .

RUN coverage run --source=src -m unittest
RUN coverage report -m