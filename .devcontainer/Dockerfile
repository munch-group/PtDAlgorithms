FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/anaconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    gdb    
    # gcc-arm-none-eabi gdb-multiarch


# RUN conda install -n base -c conda-forge -c r -y \
#         python=3.11 jupyterlab nodejs \
#         r-essentials graphviz r-rsvg r-foreach r-viridis \
#         r-devtools r-roxygen2 \
#         r-irkernel r-irdisplay r-languageserver \
#     && conda clean -afy 

RUN conda update -n base conda \
    && conda install conda-libmamba-solver \ 
    && conda config --set solver libmamba \
    && conda install -n base -c conda-forge -c r -y \
    python=3.11 jupyterlab nodejs \
    r-essentials graphviz r-rsvg r-foreach r-viridis \
    r-devtools r-roxygen2 \
    r-irkernel r-irdisplay r-languageserver \
&& conda clean -afy 

#    memfault::multi-arch-gdb 

#RUN Rscript -e 'install.packages(c("IRkernel", "IRdisplay"), repos="https://cloud.r-project.org") ; library(IRkernel) ; IRkernel::installspec()'

#IRkernel::installspec(user = FALSE)