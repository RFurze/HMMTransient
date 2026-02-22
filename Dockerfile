FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

WORKDIR /home/app

ENV DEBIAN_FRONTEND=noninteractive

# Remove MPICH (if installed) and install Open MPI
# RUN apt-get update && \
#     apt-get remove -y mpich && \
#     apt-get install -y --no-install-recommends \
#        openmpi-bin \
#        libopenmpi-dev && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# Install additional necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       software-properties-common \
       npm \
       nodejs \
       vim \
       emacs \
       cmake \
       git \
       python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Install Python packages, including ngsolve from pip
RUN pip3 install --no-cache-dir \
    tools \
    numpy \
    scipy \
    matplotlib \
    psutil \
    pytest \
    mpi4py \
    mkl \
    PyPardiso \
    pyamg 

RUN pip3 install --no-cache-dir scikit-learn pandas
# RUN pip3 install --no-cache-dir --ignore-installed torch
RUN pip3 uninstall scipy -y
RUN pip3 install scipy==1.10
RUN pip3 uninstall numpy -y
RUN pip3 install numpy==1.21.5

#     # Install petsc4py
# RUN python3 -m pip install petsc4py

# # Install ngsPETSc from GitHub
# RUN python3 -m pip install ngsPETSc

RUN pip install ngsolve

#The apt-get version of ngsolve does not have paradiso included
# RUN add-apt-repository ppa:ngsolve/ngsolve && \
    # apt-get install -y ngsolve

# ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/:/root/.local/lib/"
ENV PATH="${PATH}:/root/.local/bin"
ENV DIJITSO_CACHE_DIR=/tmp/dijitso_cache
RUN mkdir -p /tmp/dijitso_cache

COPY . /home/app
WORKDIR /home/app

CMD ["bash", "run_Transient.sh"]