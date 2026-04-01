/*
 * Sensus — Arduino Giga R1 WiFi Sensor Hub
 * ==========================================
 * All-in-one sensor node: environmental, vitals, auth, and status LEDs.
 *
 * Sensors:
 *   - DHT11:     Temperature + Humidity           (Digital pin D2)
 *   - CCS811:    CO2 + TVOC Air Quality           (I2C: SDA/SCL)
 *   - MAX30102:  Heart Rate + SpO2 ground truth   (I2C: SDA/SCL, addr 0x57)
 *   - GSR:       Galvanic Skin Response            (Analog A0)
 *   - RFID RC522: Mifare card authentication       (SPI: MOSI=D11, MISO=D12, SCK=D13, SS=D10, RST=D9)
 *   - M5Stack Finger2: Fingerprint scanner         (UART Serial1: TX=D1, RX=D0)
 *   - NeoPixel:  8-LED status strip                (Digital pin D5)
 *
 * Board: Arduino Giga R1 WiFi
 * Board Package: Arduino Mbed OS Giga Boards
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <SPI.h>

// Sensors
#include <DHT.h>
#include <Adafruit_CCS811.h>
#include <MAX30105.h>
#include <heartRate.h>
#include <Adafruit_NeoPixel.h>
#include <MFRC522.h>
#include <Adafruit_Fingerprint.h>

// ─── CONFIG ──────────────────────────────────────────────────────────────────
#define NODE_ID       "giga"
#define WIFI_SSID     "sensus-csi"
#define WIFI_PASS     "sensusAI"
#define MQTT_BROKER   "192.168.0.59"
#define MQTT_PORT     1883

// ─── PIN ASSIGNMENTS ─────────────────────────────────────────────────────────
#define DHT_PIN       2
#define DHT_TYPE      DHT11
#define GSR_PIN       A0
#define NEOPIXEL_PIN  5
#define NEOPIXEL_COUNT 8
#define RFID_SS_PIN   10
#define RFID_RST_PIN  9

// ─── TIMING ──────────────────────────────────────────────────────────────────
#define ENV_INTERVAL_MS      2000
#define HR_INTERVAL_MS       100
#define GSR_INTERVAL_MS      500
#define RFID_INTERVAL_MS     300
#define FP_INTERVAL_MS       1000
#define STATUS_INTERVAL_MS   10000
#define HR_PUBLISH_MS        1000

// ─── OBJECTS ─────────────────────────────────────────────────────────────────
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_CCS811 ccs;
MAX30105 particleSensor;
Adafruit_NeoPixel strip(NEOPIXEL_COUNT, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
MFRC522 rfid(RFID_SS_PIN, RFID_RST_PIN);
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&Serial1);

// ─── STATE ───────────────────────────────────────────────────────────────────
unsigned long lastEnv = 0, lastHR = 0, lastGSR = 0;
unsigned long lastRFID = 0, lastFP = 0, lastStatus = 0, lastHRPublish = 0;

bool ccsReady = false;
bool maxReady = false;
bool rfidReady = false;
bool fpReady = false;

// MAX30102 heart rate averaging
float beatsPerMinute = 0;
int beatAvg = 0;
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;

// Alert state for LEDs
String currentAlert = "normal";

// ─── MQTT CALLBACK ───────────────────────────────────────────────────────────
void mqttCallback(char* topic, byte* payload, unsigned int length) {
    String msg;
    for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
    String t = String(topic);

    if (t == "sensus/giga/alert") {
        currentAlert = msg;
        updateLEDs();
    }
    else if (t == "sensus/giga/leds") {
        StaticJsonDocument<64> doc;
        if (deserializeJson(doc, msg) == DeserializationError::Ok) {
            uint8_t r = doc["r"] | 0;
            uint8_t g = doc["g"] | 0;
            uint8_t b = doc["b"] | 0;
            for (int i = 0; i < NEOPIXEL_COUNT; i++) strip.setPixelColor(i, strip.Color(r, g, b));
            strip.show();
        }
    }
}

// ─── LED STATUS INDICATORS ──────────────────────────────────────────────────
void updateLEDs() {
    uint32_t color;
    if (currentAlert == "critical") {
        color = strip.Color(255, 0, 0);
    } else if (currentAlert == "warning") {
        color = strip.Color(255, 140, 0);
    } else if (currentAlert == "auth_ok") {
        color = strip.Color(0, 255, 0);
    } else {
        color = strip.Color(0, 40, 80);
    }
    for (int i = 0; i < NEOPIXEL_COUNT; i++) strip.setPixelColor(i, color);
    strip.show();
}

void ledStartupAnimation() {
    for (int i = 0; i < NEOPIXEL_COUNT; i++) {
        strip.setPixelColor(i, strip.Color(0, 100, 255));
        strip.show();
        delay(80);
    }
    delay(200);
    for (int i = 0; i < NEOPIXEL_COUNT; i++) {
        strip.setPixelColor(i, strip.Color(0, 40, 80));
    }
    strip.show();
}

// ─── WiFi ────────────────────────────────────────────────────────────────────
void connectWiFi() {
    WiFi.disconnect();
    delay(500);
    Serial.print("[WiFi] Connecting to '");
    Serial.print(WIFI_SSID);
    Serial.println("'...");
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    int timeout = 40;
    while (WiFi.status() != WL_CONNECTED && timeout > 0) {
        delay(500);
        Serial.print(".");
        timeout--;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("\n[WiFi] Connected! IP: ");
        Serial.println(WiFi.localIP());
        for (int i = 0; i < NEOPIXEL_COUNT; i++) strip.setPixelColor(i, strip.Color(0, 255, 0));
        strip.show();
        delay(300);
        updateLEDs();
    } else {
        Serial.println("\n[WiFi] FAILED");
        for (int i = 0; i < NEOPIXEL_COUNT; i++) strip.setPixelColor(i, strip.Color(255, 0, 0));
        strip.show();
    }
}

// ─── MQTT ────────────────────────────────────────────────────────────────────
void mqttReconnect() {
    if (WiFi.status() != WL_CONNECTED) return;

    String clientId = String("sensus-giga-") + String(random(1000));
    Serial.print("[MQTT] Connecting to ");
    Serial.print(MQTT_BROKER);
    Serial.println("...");

    if (mqtt.connect(clientId.c_str())) {
        Serial.println("[MQTT] Connected!");
        mqtt.subscribe("sensus/giga/alert");
        mqtt.subscribe("sensus/giga/leds");

        StaticJsonDocument<128> doc;
        doc["node"] = NODE_ID;
        doc["event"] = "connected";
        doc["ip"] = WiFi.localIP().toString();
        char buf[128];
        serializeJson(doc, buf);
        mqtt.publish("sensus/giga/status", buf);
    } else {
        Serial.print("[MQTT] Failed, rc=");
        Serial.println(mqtt.state());
    }
}

void publishJson(const char* topic, JsonDocument& doc) {
    char buf[384];
    serializeJson(doc, buf, sizeof(buf));
    mqtt.publish(topic, buf);
}

// ─── SENSOR INIT ─────────────────────────────────────────────────────────────
void initSensors() {
    dht.begin();
    Serial.println("[DHT11] Initialized on pin D2");

    if (ccs.begin()) {
        ccsReady = true;
        while (!ccs.available()) delay(100);
        Serial.println("[CCS811] Initialized (I2C)");
    } else {
        Serial.println("[CCS811] NOT FOUND - skipping");
    }

    if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        maxReady = true;
        particleSensor.setup();
        particleSensor.setPulseAmplitudeRed(0x0A);
        particleSensor.setPulseAmplitudeGreen(0);
        Serial.println("[MAX30102] Initialized (I2C)");
    } else {
        Serial.println("[MAX30102] NOT FOUND - skipping");
    }

    SPI.begin();
    rfid.PCD_Init();
    delay(100);
    rfidReady = true;
    rfid.PCD_Init();
    Serial.println("[RFID] RC522 Initialized (SPI)");

    Serial1.begin(57600);
    finger.begin(57600);
    if (finger.verifyPassword()) {
        fpReady = true;
        finger.getTemplateCount();
        Serial.print("[Fingerprint] Found, templates: ");
        Serial.println(finger.templateCount);
    } else {
        Serial.println("[Fingerprint] NOT FOUND - skipping");
    }

    strip.begin();
    strip.setBrightness(60);
    strip.show();
    Serial.println("[NeoPixel] 8-LED strip on pin D5");
}

// ─── SENSOR READING FUNCTIONS ────────────────────────────────────────────────

void readAndPublishEnv() {
    StaticJsonDocument<256> doc;
    doc["ts"] = millis();

    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    if (!isnan(temp)) doc["temp"] = serialized(String(temp, 1));
    if (!isnan(hum))  doc["humidity"] = serialized(String(hum, 1));

    if (ccsReady && ccs.available() && !ccs.readData()) {
        doc["co2"] = ccs.geteCO2();
        doc["tvoc"] = ccs.getTVOC();
    }

    publishJson("sensus/giga/env", doc);
}

void readMAX30102() {
    if (!maxReady) return;

    long irValue = particleSensor.getIR();

    if (irValue > 50000) {
        if (checkForBeat(irValue)) {
            long delta = millis() - lastBeat;
            lastBeat = millis();
            beatsPerMinute = 60.0 / (delta / 1000.0);

            if (beatsPerMinute > 30 && beatsPerMinute < 220) {
                rates[rateSpot++ % RATE_SIZE] = (byte)beatsPerMinute;
                beatAvg = 0;
                byte count = min((byte)rateSpot, RATE_SIZE);
                for (byte i = 0; i < count; i++) beatAvg += rates[i];
                beatAvg /= count;
            }
        }
    } else {
        beatsPerMinute = 0;
        beatAvg = 0;
    }
}

void publishHR() {
    if (!maxReady) return;

    StaticJsonDocument<128> doc;
    doc["ts"] = millis();
    if (beatAvg > 0) {
        doc["hr"] = beatAvg;
        doc["bpm_raw"] = serialized(String(beatsPerMinute, 1));
        doc["finger"] = true;
    } else {
        doc["hr"] = 0;
        doc["finger"] = false;
    }
    publishJson("sensus/giga/hr", doc);
}

void readAndPublishGSR() {
    int raw = analogRead(GSR_PIN);
    float voltage = raw * (3.3 / 65535.0);
    float resistance = (3.3 - voltage) / (voltage + 0.0001) * 10000.0;
    float conductance = 1000000.0 / (resistance + 1.0);

    StaticJsonDocument<128> doc;
    doc["ts"] = millis();
    doc["raw"] = raw;
    doc["conductance"] = serialized(String(conductance, 2));
    doc["voltage"] = serialized(String(voltage, 3));
    publishJson("sensus/giga/gsr", doc);
}

void checkRFID() {
    if (!rfidReady) return;
    if (!rfid.PICC_IsNewCardPresent()) return;
    if (!rfid.PICC_ReadCardSerial()) return;

    String uid = "";
    for (byte i = 0; i < rfid.uid.size; i++) {
        if (rfid.uid.uidByte[i] < 0x10) uid += "0";
        uid += String(rfid.uid.uidByte[i], HEX);
        if (i < rfid.uid.size - 1) uid += ":";
    }
    uid.toUpperCase();

    Serial.print("[RFID] Card: ");
    Serial.println(uid);

    for (int i = 0; i < NEOPIXEL_COUNT; i++) strip.setPixelColor(i, strip.Color(0, 255, 0));
    strip.show();
    delay(200);
    updateLEDs();

    StaticJsonDocument<192> doc;
    doc["ts"] = millis();
    doc["method"] = "rfid";
    doc["uid"] = uid;
    doc["type"] = rfid.PICC_GetTypeName(rfid.PICC_GetType(rfid.uid.sak));
    publishJson("sensus/giga/auth", doc);

    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
}

void checkFingerprint() {
    if (!fpReady) return;

    int result = finger.getImage();
    if (result != FINGERPRINT_OK) return;

    result = finger.image2Tz();
    if (result != FINGERPRINT_OK) return;

    result = finger.fingerSearch();
    if (result == FINGERPRINT_OK) {
        Serial.print("[Fingerprint] Match! ID: ");
        Serial.print(finger.fingerID);
        Serial.print("  Confidence: ");
        Serial.println(finger.confidence);

        for (int i = 0; i < NEOPIXEL_COUNT; i++) strip.setPixelColor(i, strip.Color(0, 255, 0));
        strip.show();
        delay(300);
        updateLEDs();

        StaticJsonDocument<128> doc;
        doc["ts"] = millis();
        doc["method"] = "fingerprint";
        doc["id"] = finger.fingerID;
        doc["confidence"] = finger.confidence;
        publishJson("sensus/giga/auth", doc);
    } else if (result == FINGERPRINT_NOTFOUND) {
        StaticJsonDocument<128> doc;
        doc["ts"] = millis();
        doc["method"] = "fingerprint";
        doc["id"] = -1;
        doc["status"] = "unknown";
        publishJson("sensus/giga/auth", doc);

        for (int i = 0; i < NEOPIXEL_COUNT; i++) strip.setPixelColor(i, strip.Color(255, 140, 0));
        strip.show();
        delay(300);
        updateLEDs();
    }
}

void publishStatus() {
    StaticJsonDocument<256> doc;
    doc["node"] = NODE_ID;
    doc["role"] = "sensor_hub";
    doc["uptime_s"] = millis() / 1000;
    doc["board"] = "Giga R1 WiFi";

    JsonArray sensors = doc.createNestedArray("sensors");
    sensors.add("dht11");
    if (ccsReady) sensors.add("ccs811");
    if (maxReady) sensors.add("max30102");
    sensors.add("gsr");
    if (rfidReady) sensors.add("rfid_rc522");
    if (fpReady) sensors.add("fingerprint");
    sensors.add("neopixel_8");

    publishJson("sensus/giga/status", doc);

    Serial.print("[Status] Up: ");
    Serial.print(millis() / 1000);
    Serial.print("s | HR: ");
    Serial.print(beatAvg);
    Serial.print(" bpm | Alert: ");
    Serial.println(currentAlert);
}

// ─── SETUP ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println();
    Serial.println("========================================");
    Serial.println("  SENSUS Sensor Hub");
    Serial.println("  Board: Arduino Giga R1 WiFi");
    Serial.println("  Sensors: DHT11 + CCS811 + MAX30102");
    Serial.println("           GSR + RFID + Fingerprint");
    Serial.println("           NeoPixel x8 Status LEDs");
    Serial.println("========================================");

    strip.begin();
    strip.setBrightness(60);
    ledStartupAnimation();

    connectWiFi();

    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setCallback(mqttCallback);
    mqtt.setBufferSize(2048);
    mqtt.setKeepAlive(30);

    initSensors();

    Serial.println("[Sensus] Giga sensor hub ready!");
}

// ─── LOOP ────────────────────────────────────────────────────────────────────
void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        connectWiFi();
        if (WiFi.status() != WL_CONNECTED) { delay(5000); return; }
    }

    if (!mqtt.connected()) {
        mqttReconnect();
        if (!mqtt.connected()) { delay(3000); return; }
    }
    mqtt.loop();

    unsigned long now = millis();

    if (now - lastEnv >= ENV_INTERVAL_MS) { readAndPublishEnv(); lastEnv = now; }
    if (now - lastHR >= HR_INTERVAL_MS) { readMAX30102(); lastHR = now; }
    if (now - lastHRPublish >= HR_PUBLISH_MS) { publishHR(); lastHRPublish = now; }
    if (now - lastGSR >= GSR_INTERVAL_MS) { readAndPublishGSR(); lastGSR = now; }
    if (now - lastRFID >= RFID_INTERVAL_MS) { checkRFID(); lastRFID = now; }
    if (now - lastFP >= FP_INTERVAL_MS) { checkFingerprint(); lastFP = now; }
    if (now - lastStatus >= STATUS_INTERVAL_MS) { publishStatus(); lastStatus = now; }
}
