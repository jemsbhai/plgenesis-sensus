#!/bin/bash
# Sensus Pi Bootstrap - Run this on the Pi after first boot
# Usage: bash bootstrap.sh

set -e
echo "=== Sensus Pi Bootstrap ==="
echo "Installing system packages..."

sudo apt-get update
sudo apt-get install -y \
    mosquitto mosquitto-clients \
    python3-pip python3-venv \
    git curl jq \
    influxdb2 \
    docker.io docker-compose \
    nginx

# Enable services
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Configure Mosquitto for local network (no auth for hackathon speed)
sudo tee /etc/mosquitto/conf.d/sensus.conf > /dev/null <<EOF
listener 1883 0.0.0.0
allow_anonymous true
EOF
sudo systemctl restart mosquitto

# Create Python venv
echo "Setting up Python environment..."
python3 -m venv ~/sensus-venv
source ~/sensus-venv/bin/activate

pip install --upgrade pip
pip install \
    paho-mqtt \
    flask flask-cors flask-socketio \
    numpy scipy \
    influxdb-client \
    google-generativeai \
    elevenlabs \
    pymongo \
    snowflake-connector-python \
    auth0-python \
    python-dotenv \
    requests \
    eventlet

# Create project directories
mkdir -p ~/sensus/{data,logs,models}

echo ""
echo "=== Bootstrap Complete ==="
echo "Next: copy the sensus service files to ~/sensus/"
echo "Then: source ~/sensus-venv/bin/activate && python ~/sensus/main.py"
