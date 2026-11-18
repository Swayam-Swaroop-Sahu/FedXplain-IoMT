# Open issues

- Raw captures, source extraction scripts, device/session keys and original benchmark split membership are absent from the workspace. Supply a capture-to-row manifest and the original extractor to reproduce the historical raw-to-feature pipeline and to evaluate independent devices. Identical-feature grouping is the available alternative.
- Original physical units and extractor formulas for several Bluetooth statistics are unrecorded.
- Authors must confirm the public release identifier, licence, author biographies, photographs, applicable declarations and any intended supervisor-specific XAI citation. The verified related work includes the supervisor's federated intrusion-detection paper.

- Exact feature grouping does not establish capture independence. Rounded signatures continue to cross the train, validation and test partitions. The group bootstrap treats exact feature groups as its resampling unit.
- A full historical-budget rerun was stopped after benchmarking throughput. The completed comparison uses six rounds, one local epoch, batch size 256 and six pooled epochs. The main suite, output diagnostics and SHAP use five seeds. The omitted-family, adaptation and sensitivity suites use seeds 42 and 7. See the experiment summaries for the parameter ranges and step budgets.
- SCAFFOLD and FedNova remain cited alternatives. Their optional empirical controls were outside the completed compute budget. No edge latency claim is retained.
- This binary detector yields attack-as-positive recall by source label and precision against the corresponding benign controls. Multiclass attack-family precision requires a separate family classifier.
