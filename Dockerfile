FROM nvcr.io/nvidia/pytorch:22.04-py3

WORKDIR /BRA/work/

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends ffmpeg

COPY ./NGC_requirements.txt /BRA/requirements.txt
RUN python -m pip install -r /BRA/requirements.txt

COPY . /BRA/

ENTRYPOINT [ "python", "work_dev.py" ]
CMD ["--help"]