# Multi-stage build for Neural Chatbot
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with proper home directory
RUN groupadd -r chatbot && useradd -r -g chatbot -d /app -s /bin/bash chatbot

# Set working directory
WORKDIR /app

# Create matplotlib config directory
RUN mkdir -p /tmp/matplotlib && chown -R chatbot:chatbot /tmp/matplotlib

# Copy and install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create NLTK download script
RUN echo 'import nltk\n\
import ssl\n\
\n\
try:\n\
    _create_unverified_https_context = ssl._create_unverified_context\n\
except AttributeError:\n\
    pass\n\
else:\n\
    ssl._create_default_https_context = _create_unverified_https_context\n\
\n\
downloads = ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]\n\
\n\
for item in downloads:\n\
    try:\n\
        nltk.download(item, download_dir="/usr/local/share/nltk_data", quiet=True)\n\
        print(f"Downloaded: {item}")\n\
    except Exception as e:\n\
        print(f"Failed to download {item}: {e}")\n\
\n\
print("NLTK downloads completed")' > /tmp/download_nltk.py

# Download NLTK data
RUN python /tmp/download_nltk.py && \
    chmod -R 755 /usr/local/share/nltk_data && \
    rm /tmp/download_nltk.py

# Copy application code
COPY src/ src/
COPY setup.py .
COPY README.md .
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create necessary directories and set permissions
RUN mkdir -p /app/models /app/logs /app/data /app/plots /app/reports /app/config && \
    chown -R chatbot:chatbot /app

# Copy configuration and data files
COPY src/neural_chatbot/config/config.yaml /app/config/config.yaml
COPY intents.json /app/intents.json

# Set correct ownership
RUN chown -R chatbot:chatbot /app

# Switch to non-root user
USER chatbot

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

# Default command
CMD ["python", "-m", "neural_chatbot.api.app"]

# Production stage
FROM base AS production

# Switch back to root to install gunicorn config
USER root

# Copy gunicorn configuration
COPY deployment/gunicorn.conf.py /app/gunicorn.conf.py

# Set ownership and switch back to chatbot user
RUN chown chatbot:chatbot /app/gunicorn.conf.py
USER chatbot

# Override CMD for production
CMD ["gunicorn", "--config", "gunicorn.conf.py", "neural_chatbot.api.app:create_app()"]

# Development stage
FROM base AS development

# Switch back to root to install dev dependencies
USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Set ownership and switch back to chatbot user
RUN chown -R chatbot:chatbot /app
USER chatbot

# Override CMD for development
CMD ["python", "-m", "neural_chatbot.api.app", "--debug"]