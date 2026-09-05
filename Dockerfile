# graphdiff in a container, built from the offline bundle so the image
# assembles without network access (only the base image is pulled).
#
#   scripts/build_offline_bundle.sh                       # on a connected machine
#   docker build --build-arg BUNDLE=dist/graphdiff-offline-0.1.0.tar.gz -t graphdiff .
#   docker run --rm -v "$PWD/graphs:/data" graphdiff compare /data/a.graphml /data/b.graphml
#   docker run --rm -p 8080:8080 -v "$PWD:/data" graphdiff serve /data/diff.html --port 8080 --no-open
#
# The runtime stage has no compiler, no pip cache and no network expectations;
# nothing in graphdiff phones home.
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS install
ARG BUNDLE
WORKDIR /bundle
COPY ${BUNDLE} bundle.tar.gz
RUN tar -xzf bundle.tar.gz --strip-components=1 \
 && python -m pip install --no-index --find-links wheelhouse --requirement requirements.txt --quiet \
 && python -m pip install --no-index --find-links wheelhouse --no-deps graphdiff --quiet \
 && python -m pip cache purge >/dev/null 2>&1 || true

FROM python:${PYTHON_VERSION}-slim
ARG PYTHON_VERSION=3.11
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY --from=install /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=install /usr/local/bin/graphdiff /usr/local/bin/graphdiff
RUN useradd --create-home --uid 1000 graphdiff
USER graphdiff
WORKDIR /data
# serve binds 127.0.0.1 by design; inside a container that means "this container".
ENTRYPOINT ["graphdiff"]
CMD ["--help"]
