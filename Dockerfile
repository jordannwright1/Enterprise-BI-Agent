FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

# Set up a new user 'user' with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-install Chromium
RUN playwright install chromium

# Copy code
COPY --chown=user . .

# Hugging Face Spaces usually looks for port 7860
CMD ["streamlit", "run", "Main.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
