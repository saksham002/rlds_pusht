# Custom Dataflow worker container for the RoboCOIN bimanual dataset builder.
# Must match the local Beam SDK version (2.69.0) and Python version (3.12).
#
# Build and push via Cloud Build (no local Docker needed):
#   gcloud builds submit \
#       --tag europe-west4-docker.pkg.dev/cmu-aidm-v2/robocoin-dataflow/worker:latest

FROM apache/beam_python3.12_sdk:2.69.0

# System dependencies for pyav (video decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pin apache-beam[gcp] to match the base image
RUN pip install --no-cache-dir "apache-beam[gcp]==2.69.0"

# Install Python dependencies (lerobot pulls torch with CUDA; override below)
RUN pip install --no-cache-dir \
    tensorflow==2.20.0 \
    tensorflow-datasets==4.9.9 \
    lerobot==0.3.3 \
    robocoin==0.1.0.1 \
    av==16.0.1 \
    huggingface_hub==1.1.6 \
    torchcodec==0.5

# Replace CUDA torch with CPU-only build (workers have no GPUs).
# Pin versions to match lerobot/torchvision constraints.
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.1+cpu torchvision==0.22.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fsspec==2025.3.0 numpy==2.2.6

# Worker environment.  Builder code is staged via setup_file pipeline option
# so no COPY needed here — only dependencies live in the image.
ENV ROBOCOIN_DOWNLOAD_ROOT=/tmp/robocoin_data/
ENV CLEANUP_IMMEDIATELY=1

# Beam SDK boot entrypoint
ENTRYPOINT ["/opt/apache/beam/boot"]
