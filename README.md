# FedXplain-IoMT

FedXplain-IoMT evaluates how federated parameter sharing changes benign-traffic recognition, attack recall and explanations across Wi-Fi, Message Queuing Telemetry Transport (MQTT), and Bluetooth medical traffic. The primary study groups identical feature vectors, fits client scalers on training data, and selects checkpoints using a separate validation partition.

## Find the revision

The revised implementation and manuscript are committed on the local `revision/ieee-access` branch. A checkout of `main` shows the earlier row-split study. Run these commands from this repository to identify your current branch and inspect the code changes.

```powershell
git branch --show-current
git log --oneline main..revision/ieee-access
git diff --stat main...revision/ieee-access -- src scripts tests config access.tex README.md
```

The guide checklist maps each request to the manuscript or experiment that addresses it, with the remaining work listed separately. [Guide checklist](docs/guide_checklist.md) · [Reviewer traceability](docs/TRACEABILITY.md)

## Five-seed grouped comparison

Each entry is the mean across the five matched seeds in the primary comparison. Macro F1 is computed per protocol and then averaged with equal weight across the three clients. Bluetooth benign recall is the fraction of its benign records classified as benign.

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

Fixed grouping uses one Wi-Fi group and one MQTT–Bluetooth group. The adaptive procedure selected three singleton groups in every run. The calibration section reports the trade-off between benign recall and attack recall and distinguishes threshold decisions from probability quality. The dashboard retains the earlier row-split results as a historical comparison. [Manuscript source](access.tex) · [Generated numerical summary](results/revision/paper_summary.json)

## From experiments to the paper

| Step | Implementation | Evidence |
|---|---|---|
| Group exact feature matches across clients and build train, validation and test partitions | [Data preparation](src/revision_data.py) | [Split summaries](results/revision/splits), [label inventory](docs/label_inventory.csv) |
| Train matching configurations and select checkpoints by validation loss | [Training](src/revision_train.py), [runner](src/run_revision.py), [configuration](config/revision.yml) | [70 primary runs](results/revision/runs), [28 family-withholding runs](results/revision_holdout/runs) |
| Evaluate calibration, explanations, transfer and training sensitivity | [Diagnostic runner](src/revision_probes.py), [sensitivity](src/revision_sensitivity.py), [adaptation](src/revision_scarcity.py) | [Experiment outputs](results/revision) |
| Recompute metrics from saved test predictions and generate the paper and README numbers | [Prediction validator](scripts/validate_revision.py), [table and figure generator](scripts/make_all_tables_figures.py) | [Per-label metrics](results/revision/per_label_metrics.csv), [claim-to-result map](docs/claim_to_result_map.csv) |

The main experiment, calibration and explanation analyses use seeds 42, 7, 123, 2024 and 31337. The sensitivity grid, target-excluded adaptation and reconnaissance-family withholding use seeds 42 and 7. The main schedule uses six rounds, one local epoch and batches of up to 256. [Configuration](config/revision.yml) · [Run counts and measured training time](results/revision/training_summary.json)

[Model diagram](figures/fig_iomt_mlp_architecture.png) · [Protocol and adaptive grouping](figures/revision_topology.png) · [Attack-label prevalence](figures/fig_attack_distribution.png) · [Sensitivity figure](figures/revision_sensitivity.png) · [Reference verification](refs/reading_notes.md)

## Data and setup

Obtain the benchmark from its official provider and follow its acquisition and use terms. The runner reads the derived Wi-Fi, MQTT and Bluetooth tables in their original column order. It retains source-family labels alongside binary training labels. Capture-independent reconstruction requires the original extractor and a capture-to-row manifest. [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html), [input tables](data/study)

The recorded environment is Windows with Python **3.14.7**, CPU PyTorch, and pinned numerical and plotting dependencies. The revision training uses one numerical thread per worker. [Environment lock](requirements.lock)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock -r requirements-dev.txt
.venv\Scripts\python.exe -m scripts.reproduce_revision
```

The reproduction entry point runs the baseline, withholding, diagnostic, sensitivity and adaptation suites, validates predictions, and regenerates manuscript tables, numerical macros and figures. Completed training configurations resume from their result files. For a clean repeat, use a fresh workspace with the derived tables or keep existing evaluation outputs in a separate directory before starting. [Archived evidence](results_v1_archive)

With the local experiment artifacts available, the following commands check the code, recompute metrics from predictions and match the generated numerical outputs to the manuscript and this README. Start the dashboard with `.venv\Scripts\python.exe -m streamlit run app.py`.

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m scripts.validate_revision
.venv\Scripts\python.exe -m scripts.make_all_tables_figures --check
```

## Outputs

| Location | Contents |
|---|---|
| [src](src) | Data preparation, training, calibration, attribution and adaptation |
| [results/revision](results/revision) | Main per-run results, diagnostics, training curves and evidence tables |
| [results/revision_holdout](results/revision_holdout) | Reconnaissance-family withholding results |
| [figures](figures) | Architecture, sensitivity, probability and detection figures |
| [refs](refs) | Bibliography, reading notes and DOI verification records |
| [docs](docs) | Traceability, feature schema, label inventory and numerical sources |

The local reproduction bundle includes model checkpoints, split manifests, prediction-level files and individual SHAP outputs. The repo tracks source files and summary results. The larger artifacts and compiled PDF are available in the local review bundles. [Regeneration checklist](docs/REPRODUCIBILITY_CHECKLIST.md) · [Verification report](docs/FINAL_REPORT.md)

## Limits and citation

Exact-feature grouping leaves capture and device independence unresolved. Possible near matches remain. The study compares binary detectors on a derived benchmark under a bounded training schedule. [Open issues](docs/OPEN_ISSUES.md)

Use the manuscript title and author list for citation. The final release identifier, licence and public archive location await author confirmation. [IEEE Access source](access.tex)
