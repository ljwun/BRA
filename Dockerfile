FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /BRA

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y -qq --no-install-recommends ffmpeg libgl1

COPY ./NGC_requirements.txt /BRA/requirements.txt
RUN python -m pip uninstall opencv -y
RUN python -m pip install -r /BRA/requirements.txt

COPY . /BRA/

ENTRYPOINT [ "python", "work/work_dev.py" ]
CMD ["--help"]
