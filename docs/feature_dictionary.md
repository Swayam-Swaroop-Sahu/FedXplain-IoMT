# Study feature dictionary

Feature order comes from the CSV headers. Missing, NaN and infinite inputs are replaced with zero before scaling. Exact units or extractor formulas require the source implementation.

| Feature | Meaning | Bluetooth handling | All zero in derived subset |
|---|---|---|---|
| Header_Length | Header length | mapped HCI statistic; exact formula unverified | None |
| Protocol Type | Extractor protocol code | mapped HCI statistic; exact formula unverified | bluetooth |
| Duration | Record duration | mapped HCI statistic; exact formula unverified | None |
| Rate | Traffic rate | mapped HCI statistic; exact formula unverified | None |
| Srate | Source traffic rate | mapped HCI statistic; exact formula unverified | None |
| Drate | Destination traffic rate | mapped HCI statistic; exact formula unverified | wifi, mqtt, bluetooth |
| fin_flag_number | FIN TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| syn_flag_number | SYN TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| rst_flag_number | RST TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| psh_flag_number | PSH TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| ack_flag_number | ACK TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| ece_flag_number | ECE TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| cwr_flag_number | CWR TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| ack_count | ACK TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| syn_count | SYN TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| fin_count | FIN TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| rst_count | RST TCP flag statistic | inapplicable; zero-filled according to extraction documentation | bluetooth |
| HTTP | HTTP protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| HTTPS | HTTPS protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| DNS | DNS protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| Telnet | Telnet protocol indicator | inapplicable; zero-filled according to extraction documentation | mqtt, bluetooth |
| SMTP | SMTP protocol indicator | inapplicable; zero-filled according to extraction documentation | mqtt, bluetooth |
| SSH | SSH protocol indicator | inapplicable; zero-filled according to extraction documentation | mqtt, bluetooth |
| IRC | IRC protocol indicator | inapplicable; zero-filled according to extraction documentation | mqtt, bluetooth |
| TCP | TCP protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| UDP | UDP protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| DHCP | DHCP protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| ARP | ARP protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| ICMP | ICMP protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| IGMP | IGMP protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| IPv | IPv protocol indicator | inapplicable; zero-filled according to extraction documentation | bluetooth |
| LLC | Logical link control indicator; Bluetooth ACL packet fraction | repurposed for ACL packet fraction | bluetooth |
| Tot sum | Total size statistic | mapped HCI statistic; exact formula unverified | None |
| Min | Minimum packet size | mapped HCI statistic; exact formula unverified | None |
| Max | Maximum packet size | mapped HCI statistic; exact formula unverified | None |
| AVG | Average packet size | mapped HCI statistic; exact formula unverified | None |
| Std | Packet size standard deviation | mapped HCI statistic; exact formula unverified | None |
| Tot size | Packet size statistic | mapped HCI statistic; exact formula unverified | None |
| IAT | Inter-arrival time | mapped HCI statistic; exact formula unverified | None |
| Number | Packet count statistic | mapped HCI statistic; exact formula unverified | None |
| Magnitue | Magnitude statistic (source spelling) | mapped HCI statistic; exact formula unverified | None |
| Radius | Dispersion radius | mapped HCI statistic; exact formula unverified | None |
| Covariance | Covariance statistic | mapped HCI statistic; exact formula unverified | None |
| Variance | Variance statistic | mapped HCI statistic; exact formula unverified | None |
| Weight | Extractor weight statistic | mapped HCI statistic; exact formula unverified | None |
