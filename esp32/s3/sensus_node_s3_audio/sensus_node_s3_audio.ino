/*
 * Sensus ESP32-S3-Sense Node Firmware
 * ====================================
 * Audio feature extraction via built-in PDM MEMS microphone.
 * Publishes energy, ZCR, spectral centroid to MQTT for health audio classification.
 *
 * Board: Seeed XIAO ESP32-S3 Sense
 * Built-in mic: MSM261D3526H1CPM (PDM)
 *   - PDM CLK: GPIO42
 *   - PDM DATA: GPIO41
 *
 * CHANGE NODE_ID FOR EACH BOARD:
 *   Board 1: "node_s3_1"
 *   Board 2: "node_s3_2"
 *   Board 3: "node_s3_3"
 *
 * Arduino IDE Settings:
 *   Board: "XIAO_ESP32S3"
 *   USB CDC On Boot: Enabled
 *   PSRAM: OPI PSRAM
 *
 * Libraries needed:
 *   - PubSubClient
 *   - ArduinoJson
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>

// ─── CHANGE THIS PER BOARD ───────────────────────────────────────────────────
#define NODE_ID       "node_s3_1"   // Change to node_s3_2, node_s3_3
// ─────────────────────────────────────────────────────────────────────────────

#define NODE_ROLE     "audio"
#define WIFI_SSID     "sensus-csi"
#define WIFI_PASS     "sensusAI"
#define MQTT_BROKER   "192.168.0.59"  // Pi Ethernet IP on router LAN
#define MQTT_PORT     1883

// ─── PDM Mic Pins (XIAO ESP32-S3 Sense built-in mic) ────────────────────────
#define I2S_PORT      I2S_NUM_0
#define I2S_SAMPLE_RATE 16000
#define I2S_BUFFER_SIZE 512
#define PDM_CLK_PIN   42
#define PDM_DATA_PIN  41

// ─── Audio Feature Config ────────────────────────────────────────────────────
#define AUDIO_PUBLISH_INTERVAL_MS  100   // Publish features every 100ms (10 Hz)
#define STATUS_PUBLISH_INTERVAL_MS 10000 // Status every 10s

// ─── MFCC-lite: 4-band energy for voice biomarkers ──────────────────────────
#define NUM_BANDS 4

WiFiClient espClient;
PubSubClient mqtt(espClient);

int16_t audio_buffer[I2S_BUFFER_SIZE];
unsigned long lastAudioPublish = 0;
unsigned long lastStatusPublish = 0;

// Rolling stats for voice biomarker estimation
float prev_energy = 0;
float energy_history[16] = {0};
int energy_idx = 0;

void setup_pdm_mic() {
    // Configure I2S in PDM RX mode for the built-in microphone
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = I2S_BUFFER_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0,
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num   = I2S_PIN_NO_CHANGE,  // Not used for PDM
        .ws_io_num    = PDM_CLK_PIN,         // PDM CLK
        .data_out_num = I2S_PIN_NO_CHANGE,   // Not used (RX only)
        .data_in_num  = PDM_DATA_PIN,        // PDM DATA
    };

    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[I2S] ERROR: Driver install failed: %d\n", err);
        return;
    }

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[I2S] ERROR: Pin config failed: %d\n", err);
        return;
    }

    i2s_zero_dma_buffer(I2S_PORT);
    Serial.println("[I2S] PDM microphone initialized (GPIO42 CLK, GPIO41 DATA)");
}

void compute_and_publish_audio() {
    size_t bytes_read = 0;
    esp_err_t result = i2s_read(I2S_PORT, audio_buffer, sizeof(audio_buffer),
                                &bytes_read, pdMS_TO_TICKS(100));

    if (result != ESP_OK || bytes_read == 0) return;

    int num_samples = bytes_read / sizeof(int16_t);
    if (num_samples < 32) return;

    // ─── Energy (RMS) ───────────────────────────────────────────────
    float sum_sq = 0;
    for (int i = 0; i < num_samples; i++) {
        float s = audio_buffer[i] / 32768.0f;
        sum_sq += s * s;
    }
    float energy = sqrtf(sum_sq / num_samples);

    // ─── Zero Crossing Rate ─────────────────────────────────────────
    int zc = 0;
    for (int i = 1; i < num_samples; i++) {
        if ((audio_buffer[i] > 0 && audio_buffer[i - 1] < 0) ||
            (audio_buffer[i] < 0 && audio_buffer[i - 1] > 0)) {
            zc++;
        }
    }
    float zcr = (float)zc / num_samples;

    // ─── 4-Band Energy (pseudo-spectral features) ───────────────────
    int quarter = num_samples / NUM_BANDS;
    float band_energy[NUM_BANDS] = {0};
    for (int b = 0; b < NUM_BANDS; b++) {
        for (int i = b * quarter; i < (b + 1) * quarter && i < num_samples; i++) {
            float s = fabsf(audio_buffer[i] / 32768.0f);
            band_energy[b] += s * s;
        }
        band_energy[b] /= quarter;
    }

    // ─── Spectral Centroid (weighted by band index) ─────────────────
    float total_e = 0;
    float weighted_e = 0;
    for (int b = 0; b < NUM_BANDS; b++) {
        total_e += band_energy[b];
        weighted_e += band_energy[b] * (b + 1);
    }
    float spectral_centroid = (total_e > 1e-10f) ? (weighted_e / total_e) / NUM_BANDS : 0.5f;

    // ─── Energy Delta (for onset detection — cough detection) ───────
    float energy_delta = energy - prev_energy;
    prev_energy = energy;

    // ─── Rolling energy variance (voice stability proxy) ────────────
    energy_history[energy_idx % 16] = energy;
    energy_idx++;
    float e_mean = 0, e_var = 0;
    int hist_count = min(energy_idx, 16);
    for (int i = 0; i < hist_count; i++) e_mean += energy_history[i];
    e_mean /= hist_count;
    for (int i = 0; i < hist_count; i++) {
        float d = energy_history[i] - e_mean;
        e_var += d * d;
    }
    e_var /= hist_count;

    // ─── Peak amplitude ─────────────────────────────────────────────
    int16_t peak = 0;
    for (int i = 0; i < num_samples; i++) {
        if (abs(audio_buffer[i]) > peak) peak = abs(audio_buffer[i]);
    }
    float peak_norm = peak / 32768.0f;

    // ─── Publish JSON ───────────────────────────────────────────────
    StaticJsonDocument<384> doc;
    doc["ts"] = millis();
    doc["energy"] = serialized(String(energy, 4));
    doc["zcr"] = serialized(String(zcr, 4));
    doc["spectral_centroid"] = serialized(String(spectral_centroid, 4));
    doc["energy_delta"] = serialized(String(energy_delta, 4));
    doc["energy_var"] = serialized(String(e_var, 6));
    doc["peak"] = serialized(String(peak_norm, 4));

    // Band energies
    JsonArray bands = doc.createNestedArray("bands");
    for (int b = 0; b < NUM_BANDS; b++) {
        bands.add(serialized(String(band_energy[b], 6)));
    }

    char buf[384];
    serializeJson(doc, buf, sizeof(buf));

    String topic = String("sensus/") + NODE_ID + "/audio";
    mqtt.publish(topic.c_str(), buf);
}

void mqtt_reconnect() {
    if (WiFi.status() != WL_CONNECTED) return;  // Don't try MQTT without WiFi

    String clientId = String("sensus-") + NODE_ID + "-" + String(random(1000));
    Serial.printf("[MQTT] Connecting to %s:%d as %s...\n",
                  MQTT_BROKER, MQTT_PORT, clientId.c_str());

    if (mqtt.connect(clientId.c_str())) {
        Serial.println("[MQTT] Connected!");

        // Publish online status
        StaticJsonDocument<128> doc;
        doc["node"] = NODE_ID;
        doc["role"] = NODE_ROLE;
        doc["event"] = "connected";
        doc["ip"] = WiFi.localIP().toString();
        char buf[128];
        serializeJson(doc, buf, sizeof(buf));
        String topic = String("sensus/") + NODE_ID + "/status";
        mqtt.publish(topic.c_str(), buf);
    } else {
        Serial.printf("[MQTT] Failed (rc=%d)\n", mqtt.state());
    }
}

void connectWiFi() {
    // CRITICAL: fully tear down any existing connection first
    WiFi.disconnect(true);  // true = also turn off WiFi
    delay(1000);            // let the radio fully stop

    WiFi.mode(WIFI_STA);
    delay(100);

    Serial.printf("[WiFi] Connecting to '%s'...\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    // Wait up to 15 seconds
    int timeout = 30;
    while (WiFi.status() != WL_CONNECTED && timeout > 0) {
        delay(500);
        Serial.print(".");
        timeout--;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected! IP: %s  RSSI: %d dBm\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
    } else {
        Serial.printf("\n[WiFi] FAILED (status=%d). Will retry in 5s...\n", WiFi.status());
    }
}

void setup() {
    Serial.begin(115200);
    delay(2000);  // Give USB CDC time to enumerate

    Serial.println();
    Serial.println("========================================");
    Serial.println("  SENSUS Audio Node");
    Serial.printf( "  ID: %s\n", NODE_ID);
    Serial.println("  Board: XIAO ESP32-S3 Sense");
    Serial.println("  Mic: PDM (built-in)");
    Serial.println("========================================");

    // Connect to WiFi
    connectWiFi();

    // Setup MQTT
    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setBufferSize(2048);
    mqtt.setKeepAlive(30);

    // Setup PDM microphone
    setup_pdm_mic();

    Serial.printf("[Sensus] Ready — MQTT: %s:%d  Topic: sensus/%s/audio\n",
                  MQTT_BROKER, MQTT_PORT, NODE_ID);
}

void loop() {
    // WiFi health check — reconnect if dropped
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Lost connection, reconnecting...");
        connectWiFi();
        if (WiFi.status() != WL_CONNECTED) {
            delay(5000);  // Back off before retry
            return;
        }
    }

    // MQTT health check
    if (!mqtt.connected()) {
        mqtt_reconnect();
        if (!mqtt.connected()) {
            delay(3000);
            return;
        }
    }
    mqtt.loop();

    unsigned long now = millis();

    // Audio features at 10 Hz
    if (now - lastAudioPublish >= AUDIO_PUBLISH_INTERVAL_MS) {
        compute_and_publish_audio();
        lastAudioPublish = now;
    }

    // Status heartbeat every 10s
    if (now - lastStatusPublish >= STATUS_PUBLISH_INTERVAL_MS) {
        StaticJsonDocument<192> doc;
        doc["node"] = NODE_ID;
        doc["role"] = NODE_ROLE;
        doc["uptime_s"] = now / 1000;
        doc["heap_free"] = ESP.getFreeHeap();
        doc["wifi_rssi"] = WiFi.RSSI();
        doc["ip"] = WiFi.localIP().toString();
        doc["mic"] = "pdm_builtin";

        char buf[192];
        serializeJson(doc, buf, sizeof(buf));

        String topic = String("sensus/") + NODE_ID + "/status";
        mqtt.publish(topic.c_str(), buf);

        Serial.printf("[Status] Heap: %u | RSSI: %d dBm | Uptime: %lus\n",
                      ESP.getFreeHeap(), WiFi.RSSI(), now / 1000);

        lastStatusPublish = now;
    }
}
