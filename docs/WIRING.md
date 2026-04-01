# Sensus Wiring Guide

## Node Assignment

| Node | Board | Role | Sensors | Chair Position |
|------|-------|------|---------|----------------|
| node_1 | ESP32-C6 #1 | CSI Primary (AP) | DHT11 + CCS811 | Headrest (behind head) |
| node_2 | ESP32-C6 #2 | CSI Receiver | GSR sensor | Left armrest |
| node_3 | ESP32-C6 #3 | CSI Receiver | MAX30102 | Right armrest |
| node_s3_1 | ESP32-S3 Sense #1 | Audio + Camera | Built-in mic + OV2640 | Front-facing (1m away on tripod) |
| node_s3_2 | ESP32-S3 Sense #2 | Audio backup | Built-in mic | Under seat |
| aggregator | Raspberry Pi 5 | Fusion + Dashboard | USB mic backup | Side table |

## ESP32-C6 Node 1 (Headrest) - CSI + Environmental

```
ESP32-C6          DHT11
--------          -----
GPIO4  --------  DATA
3.3V   --------  VCC
GND    --------  GND

ESP32-C6          CCS811
--------          ------
GPIO6 (SDA) ---  SDA
GPIO7 (SCL) ---  SCL
3.3V   --------  VCC
GND    --------  GND
GND    --------  WAK (tie to GND to wake)
```

## ESP32-C6 Node 2 (Left Armrest) - CSI + GSR

```
ESP32-C6          GSR Sensor
--------          ----------
GPIO0 (ADC) ---  SIG (analog out)
3.3V   --------  VCC
GND    --------  GND
```

## ESP32-C6 Node 3 (Right Armrest) - CSI + MAX30102

```
ESP32-C6          MAX30102
--------          --------
GPIO6 (SDA) ---  SDA
GPIO7 (SCL) ---  SCL
3.3V   --------  VIN
GND    --------  GND
```
Note: MAX30102 I2C address is 0x57. Subject places finger on sensor during demo.

## ESP32-S3 Sense (Front-facing) - Audio + Camera

No external wiring needed. Built-in MEMS mic and OV2640 camera.
Just power via USB-C.

## Raspberry Pi 5

- Power: Official 27W USB-C PSU
- Network: WiFi to TP-Link AX1500 (sensus-csi)
- USB: Mini USB mic plugged into USB-A port
- Display: Mini HDMI screen via HDMI-D to HDMI-A cable
- Input: Rii K01XI USB dongle

## TP-Link AX1500 Router

- NOT connected to internet
- SSID: sensus-csi
- Password: sensus2026
- Channel: 1 (fixed)
- Band: 2.4 GHz only
- Width: 20 MHz

## Power Distribution

All nodes powered via USB-C from the power strip.
Pi 5 uses its dedicated 27W PSU.
Router uses its included power adapter.
