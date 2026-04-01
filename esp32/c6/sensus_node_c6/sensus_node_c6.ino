/*
 * Sensus ESP32-C6 Node Firmware
 * WiFi CSI extraction + Environmental sensors + MQTT
 * Auto-reconnects WiFi and MQTT if connection drops.
 *
 * CHANGE NODE_ID PER BOARD: node_1, node_2, node_3
 * Board: XIAO_ESP32C6, USB CDC On Boot: Enabled
 */

#include <WiFi.h>
#include "esp_wifi.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>

// ─── CHANGE THIS PER BOARD ──────────────────────────────────────────────────
#define NODE_ID       "node_1"   // Change to node_2, node_3
// ────────────────────────────────────────────────────────────────────────────

#define NODE_ROLE     "csi"
#define WIFI_SSID     "sensus-csi"
#define WIFI_PASS     "sensusAI"
#define MQTT_BROKER   "192.168.0.59"
#define MQTT_PORT     1883

#define I2C_SDA       6
#define I2C_SCL       7
#define GSR_PIN       0

WiFiClient espClient;
PubSubClient mqtt(espClient);

unsigned long lastEnvPublish = 0;
unsigned long lastStatusPublish = 0;
unsigned long lastCSIPublish = 0;

#define CSI_BUFFER_SIZE 64
int16_t csi_amplitude[CSI_BUFFER_SIZE];
int8_t csi_phase_i[CSI_BUFFER_SIZE];
int8_t csi_phase_q[CSI_BUFFER_SIZE];
int csi_num_subs = 0;
volatile bool csi_data_ready = false;

void wifi_csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len == 0) return;
    int len = info->len;
    int8_t *buf = (int8_t *)info->buf;
    int num_pairs = len / 2;
    if (num_pairs > CSI_BUFFER_SIZE) num_pairs = CSI_BUFFER_SIZE;
    for (int i = 0; i < num_pairs; i++) {
        int8_t re = buf[2 * i];
        int8_t im = buf[2 * i + 1];
        // Proper magnitude: sqrt(re^2 + im^2) — approximated with integer math
        csi_amplitude[i] = (int16_t)sqrt((float)(re * re + im * im));
        csi_phase_i[i] = re;
        csi_phase_q[i] = im;
    }
    csi_num_subs = num_pairs;
    csi_data_ready = true;
}

void setup_csi() {
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_csi(true);
    esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL);
    Serial.println("[CSI] Enabled with promiscuous mode");
}

void connectWiFi() {
    WiFi.disconnect(true);
    delay(500);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("[WiFi] Connecting");
    int timeout = 40;
    while (WiFi.status() != WL_CONNECTED && timeout > 0) {
        delay(500);
        Serial.print(".");
        timeout--;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("\n[WiFi] IP: ");
        Serial.println(WiFi.localIP());
        // Re-enable CSI after WiFi reconnect
        setup_csi();
    } else {
        Serial.println("\n[WiFi] FAILED");
    }
}

void mqtt_reconnect() {
    if (WiFi.status() != WL_CONNECTED) return;

    String clientId = String("sensus-") + NODE_ID + "-" + String(random(1000));
    Serial.print("[MQTT] Connecting...");
    if (mqtt.connect(clientId.c_str())) {
        Serial.println("connected");
        StaticJsonDocument<128> doc;
        doc["node"] = NODE_ID;
        doc["status"] = "online";
        doc["role"] = NODE_ROLE;
        doc["ip"] = WiFi.localIP().toString();
        char buf[128];
        serializeJson(doc, buf);
        String topic = String("sensus/") + NODE_ID + "/status";
        mqtt.publish(topic.c_str(), buf);
    } else {
        Serial.print("failed rc=");
        Serial.println(mqtt.state());
    }
}

void publish_csi() {
    if (!csi_data_ready || csi_num_subs == 0) return;
    csi_data_ready = false;

    StaticJsonDocument<2048> doc;
    doc["ts"] = millis();
    doc["n"] = csi_num_subs;

    // Amplitude array
    JsonArray amp = doc.createNestedArray("amplitude");
    for (int i = 0; i < csi_num_subs; i++) amp.add(csi_amplitude[i]);

    // Phase as atan2(Q, I) scaled to int for efficiency
    JsonArray phase = doc.createNestedArray("phase");
    for (int i = 0; i < csi_num_subs; i++) {
        float ph = atan2((float)csi_phase_q[i], (float)csi_phase_i[i]);
        phase.add(serialized(String(ph, 3)));
    }

    char buf[2048];
    size_t len = serializeJson(doc, buf, sizeof(buf));
    if (len > 0) {
        String topic = String("sensus/") + NODE_ID + "/csi";
        mqtt.publish(topic.c_str(), buf);
    }
}

void publish_env() {
    StaticJsonDocument<256> doc;
    doc["ts"] = millis();
    doc["temp"] = 0;
    doc["humidity"] = 0;
    doc["co2"] = 0;
    doc["tvoc"] = 0;
    char buf[256];
    serializeJson(doc, buf);
    String topic = String("sensus/") + NODE_ID + "/env";
    mqtt.publish(topic.c_str(), buf);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println();
    Serial.println("========================================");
    Serial.print("  SENSUS CSI Node: ");
    Serial.println(NODE_ID);
    Serial.println("  Board: XIAO ESP32-C6");
    Serial.println("========================================");

    Wire.begin(I2C_SDA, I2C_SCL);
    connectWiFi();

    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setBufferSize(2048);
    analogReadResolution(12);
    Serial.println("[Sensus] Node ready");
}

void loop() {
    // WiFi auto-reconnect
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Lost, reconnecting...");
        connectWiFi();
        if (WiFi.status() != WL_CONNECTED) {
            delay(5000);
            return;
        }
    }

    // MQTT auto-reconnect (no give-up limit)
    if (!mqtt.connected()) {
        mqtt_reconnect();
        if (!mqtt.connected()) {
            delay(3000);
            return;
        }
    }
    mqtt.loop();

    unsigned long now = millis();

    // CSI at 10 Hz
    if (now - lastCSIPublish > 100) {
        publish_csi();
        lastCSIPublish = now;
    }

    // Env at 1 Hz
    if (now - lastEnvPublish > 1000) {
        publish_env();
        lastEnvPublish = now;
    }

    // Status heartbeat at 0.1 Hz
    if (now - lastStatusPublish > 10000) {
        StaticJsonDocument<128> doc;
        doc["node"] = NODE_ID;
        doc["uptime_s"] = now / 1000;
        doc["heap_free"] = ESP.getFreeHeap();
        doc["wifi_rssi"] = WiFi.RSSI();
        char buf[128];
        serializeJson(doc, buf);
        String topic = String("sensus/") + NODE_ID + "/status";
        mqtt.publish(topic.c_str(), buf);
        lastStatusPublish = now;
    }
}
