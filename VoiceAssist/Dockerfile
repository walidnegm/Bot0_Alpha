# Use Python 3.10 slim image
FROM python:3.10-slim

# Install necessary packages for audio and building PyAudio
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libasound2 \
    libpulse0 \
    ffmpeg \
    && apt-get clean

# Install Go
RUN wget https://golang.org/dl/go1.21.1.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.1.linux-amd64.tar.gz

# Add Go to PATH
ENV PATH="/usr/local/go/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements2.txt .
RUN pip install --no-cache-dir -r requirements2.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the application port
EXPOSE 8080

# Set the default command to open a shell
CMD ["bash"]
