# FedXplain-IoMT

FedXplain-IoMT evaluates how federated parameter sharing changes benign-traffic recognition, attack recall and explanations across Wi-Fi, Message Queuing Telemetry Transport (MQTT), and Bluetooth medical traffic. The primary study groups identical feature vectors, fits client scalers on training data, and selects checkpoints using a separate validation partition.

## Observed comparison

<!-- revision-summary:start -->

| Method | Client-mean macro F1 | Bluetooth benign recall |
|---|---:|---:|
| FedAvg | 0.6376 | 0.0161 |
| FedPer | 0.8726 | 0.7180 |
| Fixed groups | 0.9604 | 0.9827 |
| Adaptive groups | 0.9634 | 0.9944 |
| Equal client weights | 0.9097 | 0.9592 |
| Class-balanced loss | 0.7116 | 0.2478 |
| Local-only | 0.9626 | 0.9965 |
| Centralized | 0.9639 | 0.9963 |
<!-- revision-summary:end -->

The manuscript gives the calibration, representation, training-budget, family-withholding, and adaptation results. The dashboard also provides the historical row-split study for comparison. [Manuscript source](access.tex), [revision evidence](results/revision/paper_summary.json)

## Data and setup

Obtain the benchmark from its official provider and follow its acquisition and use terms. The runner reads the derived Wi-Fi, MQTT and Bluetooth tables in their original column order. It retains source-family labels alongside binary training labels. Capture-independent reconstruction requires the original extractor and a capture-to-row manifest. [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html), [input tables](data/study)

The recorded environment is Windows with Python **3.14.7**, CPU PyTorch, and pinned numerical and plotting dependencies. The revision training uses one numerical thread per worker. [Environment lock](requirements.lock)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock -r requirements-dev.txt
.venv\Scripts\python.exe -m scripts.reproduce_revision
```

The reproduction entry point runs the baseline, withholding, diagnostic, sensitivity and adaptation suites, validates predictions, and regenerates manuscript tables, numerical macros and figures. Completed training configurations resume from their result files. For a clean repeat, use a fresh workspace with the derived tables or keep existing evaluation outputs in a separate directory before starting. [Archived evidence](results_v1_archive)

The matched seeds and training schedules are recorded in the configuration. The main and withheld-family suites each took minutes on the reference CPU with concurrent jobs. SHAP, sensitivity and adaptation add several minutes. Runtime depends on cores and memory. [Configuration](config/revision.yml), [measured runtimes and step counts](results/revision/training_summary.json)

## Outputs

| Location | Contents |
|---|---|
| [src](src) | Data preparation, training, calibration, attribution and adaptation |
| [results/revision](results/revision) | Main per-run results, diagnostics, training curves and evidence tables |
| [results/revision_holdout](results/revision_holdout) | Reconnaissance-family withholding results |
| [figures](figures) | Architecture, sensitivity, probability and detection figures |
| [refs](refs) | Bibliography, reading notes and DOI verification records |
| [docs](docs) | Traceability, feature schema, label inventory and numerical sources |

The local reproduction bundle includes model checkpoints, split manifests, prediction-level files and individual SHAP outputs. [Regeneration checklist](docs/REPRODUCIBILITY_CHECKLIST.md)

## Limits and citation

Exact-feature grouping leaves capture and device independence unresolved. Possible near matches remain. The study compares binary detectors on a derived benchmark under a bounded training schedule. [Open issues](docs/OPEN_ISSUES.md)

Use the manuscript title and author list for citation. The final release identifier, licence and public archive location await author confirmation. [IEEE Access source](access.tex)
