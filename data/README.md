# FedXplain-IoMT Dataset Documentation

## Source

**CICIoMT2024** — Canadian Institute for Cybersecurity, Internet of Medical Things 2024 dataset.
- 40 IoMT devices across 3 protocols (Wi-Fi, MQTT, Bluetooth)
- 18 attack types across 5 categories (DDoS, DoS, Recon, MQTT-specific, Spoofing)
- Original dataset: ~8.8 million rows, ~2.2 GB of CSV data (plus ~30 GB of raw pcap files)

## Extracted Datasets

All datasets were created via **stratified random sampling** (seed=42) from the original
CICIoMT2024 CSV files. Labels were derived from source filenames (e.g., `TCP_IP-DDoS-ICMP3_train.pcap.csv` → label `TCP_IP-DDoS-ICMP`). Multiple sub-captures of the same attack type (e.g., ICMP1–ICMP8) were merged into a single class.

### Protocol Groups

The original dataset combines Wi-Fi and MQTT traffic into one CSV set. Bluetooth data exists
only as raw `.pcap` captures with no pre-extracted features, so it is **not included** in these
sampled datasets.

- **WiFi**: TCP/IP-based attacks (DDoS, DoS via ICMP/SYN/TCP/UDP), ARP Spoofing, Reconnaissance, plus Benign traffic
- **MQTT**: MQTT-specific attacks (DDoS/DoS Connect/Publish Flood, Malformed Data), plus Benign traffic

Benign traffic is shared across both protocol groups (sampled independently for each).

### POC Datasets (`data/poc/`)

Small datasets for rapid pipeline development and testing.

| File | Protocol | Rows | Classes | Features | Size |
|------|----------|------|---------|----------|------|
| `wifi_poc.csv` | WiFi | 4,995 | 14 | 45 + label | 1.3 MB |
| `mqtt_poc.csv` | MQTT | 2,998 | 6 | 45 + label | 1.0 MB |

### Study Datasets (`data/study/`)

Paper-level experiment datasets, sized for 2-3 months of federated learning research.

| File | Protocol | Rows | Classes | Features | Size |
|------|----------|------|---------|----------|------|
| `wifi_study.csv` | WiFi | 99,993 | 14 | 45 + label | 26.4 MB |
| `mqtt_study.csv` | MQTT | 49,997 | 6 | 45 + label | 15.9 MB |

### Column Schema

- **45 network flow features**: Header_Length, Protocol Type, Duration, Rate, Srate, Drate, TCP flags (fin/syn/rst/psh/ack/ece/cwr), flag counts, protocol indicators (HTTP, HTTPS, DNS, etc.), statistical features (Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Magnitude, Radius, Covariance, Variance, Weight)
- **1 label column**: `label` — attack class name or "Benign"

### Sampling Strategy

- **Proportional stratified sampling**: Each attack class is represented proportionally to its frequency in the original dataset
- **Minimum guarantee**: Small classes get at least a minimum allocation to ensure representation
- **Reproducible**: All sampling uses `numpy.random.RandomState(seed=42)`

### Attack Classes

#### WiFi (14 classes)
| Class | Category | Original Count | Study Count |
|-------|----------|---------------|-------------|
| Benign | — | 230,339 | 3,614 |
| ARP_Spoofing | Spoofing | 17,791 | 1,584 |
| Recon-OS_Scan | Reconnaissance | 20,666 | 1,611 |
| Recon-Ping_Sweep | Reconnaissance | 926 | 926 |
| Recon-Port_Scan | Reconnaissance | 106,603 | 2,432 |
| Recon-VulScan | Reconnaissance | 3,207 | 1,444 |
| TCP_IP-DDoS-ICMP | DDoS | 1,887,175 | 19,440 |
| TCP_IP-DDoS-SYN | DDoS | 974,359 | 10,721 |
| TCP_IP-DDoS-TCP | DDoS | 987,063 | 10,842 |
| TCP_IP-DDoS-UDP | DDoS | 1,998,026 | 20,498 |
| TCP_IP-DoS-ICMP | DoS | 514,724 | 6,330 |
| TCP_IP-DoS-SYN | DoS | 540,498 | 6,577 |
| TCP_IP-DoS-TCP | DoS | 462,480 | 5,831 |
| TCP_IP-DoS-UDP | DoS | 704,503 | 8,143 |

#### MQTT (6 classes)
| Class | Category | Original Count | Study Count |
|-------|----------|---------------|-------------|
| Benign | — | 230,339 | 18,389 |
| MQTT-DDoS-Connect_Flood | DDoS | 214,952 | 17,264 |
| MQTT-DDoS-Publish_Flood | DDoS | 36,039 | 4,179 |
| MQTT-DoS-Connect_Flood | DoS | 15,904 | 2,707 |
| MQTT-DoS-Publish_Flood | DoS | 52,881 | 5,411 |
| MQTT-Malformed_Data | MQTT-specific | 6,877 | 2,047 |

## Notes

- **Bluetooth data** is NOT included — the CICIoMT2024 release only provides raw `.pcap` captures for Bluetooth, with no pre-extracted CSV features. Feature extraction from Bluetooth pcaps would require a separate pipeline (e.g., using CICFlowMeter or similar tools).
- The `data/_extract_temp/` directory contains the full extraction workspace and is gitignored.
- The original `CICIoMT2024.tar.xz.zip` archive (~10.3 GB) is also gitignored.
