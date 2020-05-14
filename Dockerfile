FROM rbarile17/framework_dependencies

COPY src ./src
COPY test ./test

ENV PYTHONPATH "${PYTHONPATH}:./src"

RUN chmod a+x .

RUN coverage run --source=src -m unittest
RUN coverage report -m
