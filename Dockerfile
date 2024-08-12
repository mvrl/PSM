FROM daskdev/dask

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        subversion \
        wget \
        g++-11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

ENV PATH /opt/conda/bin:$PATH

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate sat2audio" >> ~/.bashrc && \
    /opt/conda/bin/conda clean -afy

RUN echo $(pwd)

COPY sat2audio.yml /sat2sound/

RUN cd /sat2sound && conda env create -f sat2audio.yml;

RUN chmod -R 777 /sat2sound

SHELL ["/bin/bash", "--login", "-c"]                             
CMD [ "/bin/bash" ]


