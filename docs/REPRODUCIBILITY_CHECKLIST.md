# Reproducibility checklist

- Use the pinned environment in requirements.lock and the test extras in requirements-dev.txt. The recorded machine used CPU PyTorch under Windows with Python 3.14.7.
- Run python -m scripts.reproduce_revision from the repository root with the derived feature CSVs. The numerical thread limits and deterministic PyTorch flags are set by the runner. Configuration, seeds, grouping thresholds, training budgets and SHAP sizes are in config/revision.yml.
- Match split manifests to the CSV row indices and per-seed SHA-256 digests. Preserve extractor-provided source groups when they are available.
- Verify each selected round is the minimum client-mean validation loss and compare recorded updates to the compute-matched comparator. Prediction files hold sample IDs, targets, scores, labels and the feature-group ID.
- Rebuild paper tables, macros and figures with python -m scripts.make_all_tables_figures. The --check switch compares the generated LaTeX, numerical summary and README. Claim provenance is in claim_to_result_map.csv.
- Open the Overleaf archive and run pdfLaTeX, BibTeX, then pdfLaTeX twice or use Overleaf's default LaTeX build. The source package includes the IEEE Access class and assets.
- The local release bundle includes all prediction-level outputs and checkpoints. Use the benchmark provider to acquire traffic material. Candidate release terms, persistent public archive and author declarations are pending.
