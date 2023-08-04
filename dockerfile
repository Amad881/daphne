FROM pytorchlightning/pytorch_lightning
# FROM continuumio/miniconda3
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.13-cuda12.0.1

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

WORKDIR /app
COPY lightning_rnn.py ./
COPY main.py ./
COPY models_pairwise.py ./
COPY pairwiseModel.py ./
COPY rnnModel_dot.py ./
COPY rnnModel_qkv.py ./
COPY util.py ./
COPY dataProcess.py ./
COPY data/ ./data/
COPY modelStore/ ./modelStore/
COPY environment.yml ./

# RUN conda env create -f environment.yml
# RUN echo "source activate hugLight" > ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]
# SHELL ["conda", "run", "-n", "hugLight", "/bin/bash", "-c"]
# RUN conda activate hugLight

RUN pip install fastapi 
RUN pip install uvicorn 
RUN pip install faiss-cpu
RUN pip install scikit-learn 
RUN pip install transformers 
RUN pip install pandas
RUN pip install matplotlib

RUN export PYTHONPATH=$PWD

RUN python3 main.py

CMD ["main.lambda_handler"]