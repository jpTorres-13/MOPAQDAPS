# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Prevent Python from buffering stdout/stderr (helps with logging)
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by librosa, torchaudio, etc.
# - git (if you later want to pull from a git repo)
# - build-essential (for any C-extensions that might need building)
# - libsndfile1 (for soundfile support used by librosa)
# - ffmpeg (for audio decoding, if needed)
# - wget, ca-certificates (for downloading/testing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 \
      ffmpeg \
      git \
      wget \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set a working directory inside the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# If your package is located in a subfolder 'pitch_eval', adjust accordingly.
# We assume the following file structure:
# .
# ├── pitch_eval/
# │   ├── __init__.py
# │   └── main.py
# ├── setup.py
# ├── requirements.txt
# └── (other files)
#
# This next command will install your package into the container (editable mode).
RUN pip install .

# Expose any port if your application serves over HTTP (not required here)
# EXPOSE 8000

# By default, run the CLI. The entrypoint is the 'mopaqdaps' console script
ENTRYPOINT ["mopaqdaps"]
CMD ["--help"]
