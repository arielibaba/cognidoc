FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies for PDF processing and LibreOffice
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        libreoffice-core \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install dependencies (without dev extras)
RUN uv sync --no-dev --extra ui --extra cloud --extra conversion

# Copy remaining files
COPY config/ config/
COPY Makefile ./

# Create data directories
RUN mkdir -p data/sources data/pdfs data/images data/chunks data/indexes data/vector_store data/cache

EXPOSE 7860

ENV COGNIDOC_PROJECT_DIR=/app

CMD ["uv", "run", "cognidoc", "serve", "--port", "7860"]
