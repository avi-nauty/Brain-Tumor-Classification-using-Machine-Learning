FROM python:3.12-slim

# Gradio defaults to localhost only; bind to all interfaces on the port Cloud Run sends traffic to
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install libraries first so this slow layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs (reference/ stays out of the image)
COPY inference.py app.py ./
COPY artifacts/ artifacts/
COPY examples/ examples/

# Download ImageNet weights at build time and run the artifact shape asserts,
# so cold starts are fast and a bad file fails the build instead of production
RUN python -c "import inference"

CMD ["python", "app.py"]