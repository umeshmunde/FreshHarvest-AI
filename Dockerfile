FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal OS packages required by OpenCV and imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (Standard for Hugging Face Spaces & Production)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies (CPU-optimized PyTorch to minimize container size)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY --chown=user:user . .

# Expose default port (7860 for Hugging Face Spaces, 5000/dynamic for Render)
EXPOSE 7860
EXPOSE 5000

CMD sh -c "gunicorn --workers=2 --threads=2 --timeout=120 --bind=0.0.0.0:${PORT:-7860} app:app"

