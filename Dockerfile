# Multi-stage build for Neural Chatbot
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r chatbot && useradd -r -g chatbot chatbot

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data as root user to avoid permission issues
RUN python -c "import nltk; \
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data', quiet=True); \
    nltk.download('wordnet', download_dir='/usr/local/share/nltk_data', quiet=True); \
    nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data', quiet=True); \
    nltk.download('stopwords', download_dir='/usr/local/share/nltk_data', quiet=True)" && \
    chmod -R 755 /usr/local/share/nltk_data

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

# Copy gunicorn configuration (create it if it doesn't exist)
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