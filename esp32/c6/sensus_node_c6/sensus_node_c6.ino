/*
 * Sensus ESP32-C6 Node Firmware
 * WiFi CSI extraction + Environmental sensors + MQTT
 */

#include <WiFi.h>
#include "esp_wifi.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>

#define NODE_ID       "node_2"
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
        csi_amplitude[i] = (int16_t)(abs(re) + abs(im));
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

void mqtt_reconnect() {
    int attempts = 0;
    while (!mqtt.connected() && attempts < 5) {
        Serial.print("[MQTT] Connecting...");
        String clientId = String("sensus-") + NODE_ID;
        if (mqtt.connect(clientId.c_str())) {
            Serial.println("connected");
            StaticJsonDocument<128> doc;
            doc["node"] = NODE_ID;
            doc["status"] = "online";
            doc["role"] = NODE_ROLE;
            char buf[128];
            serializeJson(doc, buf);
            String topic = String("sensus/") + NODE_ID + "/status";
            mqtt.publish(topic.c_str(), buf);
            return;
        }
        Serial.print("failed rc=");
        Serial.println(mqtt.state());
        attempts++;
        delay(2000);
    }
}

void publish_csi() {
    if (!csi_data_ready || csi_num_subs == 0) return;
    csi_data_ready = false;
    StaticJsonDocument<1536> doc;
    doc["ts"] = millis();
    doc["n"] = csi_num_subs;
    JsonArray amp = doc.createNestedArray("amplitude");
    for (int i = 0; i < csi_num_subs; i++) amp.add(csi_amplitude[i]);
    char buf[1536];
    serializeJson(doc, buf);
    String topic = String("sensus/") + NODE_ID + "/csi";
    mqtt.publish(topic.c_str(), buf);
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

void publish_gsr() {
    int raw = analogRead(GSR_PIN);
    float voltage = raw * (3.3 / 4095.0);
    float conductance = voltage / 10000.0 * 1000000.0;
    StaticJsonDocument<128> doc;
    doc["ts"] = millis();
    doc["raw"] = raw;
    doc["conductance"] = conductance;
    char buf[128];
    serializeJson(doc, buf);
    String topic = String("sensus/") + NODE_ID + "/gsr";
    mqtt.publish(topic.c_str(), buf);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== Sensus Node " NODE_ID " ===");
    Wire.begin(I2C_SDA, I2C_SCL);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("[WiFi] Connecting");
    int wifi_attempts = 0;
    while (WiFi.status() != WL_CONNECTED && wifi_attempts < 40) {
        delay(500);
        Serial.print(".");
        wifi_attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println();
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\n[WiFi] FAILED to connect!");
    }
    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setBufferSize(2048);
    setup_csi();
    analogReadResolution(12);
    Serial.println("[Sensus] Node ready");
}

void loop() {
    if (!mqtt.connected()) mqtt_reconnect();
    mqtt.loop();
    unsigned long now = millis();
    if (now - lastCSIPublish > 100) {
        publish_csi();
        lastCSIPublish = now;
    }
    if (now - lastEnvPublish > 1000) {
        publish_env();
        publish_gsr();
        lastEnvPublish = now;
    }
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
