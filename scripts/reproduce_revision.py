"""Run the evaluation workflow from the repository root with the pinned environment."""
import os
from pathlib import Path
import subprocess
import sys


def run(*arguments):
    print(" ".join(arguments),flush=True)
    subprocess.run([sys.executable,"-m",*arguments],check=True,env={
        **os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})


def main():
    run("src.run_revision")
    family=["Recon-OS_Scan","Recon-Ping_Sweep","Recon-Port_Scan","Recon-VulScan"]
    args=[v for label in family for v in ("--exclude-label",label)]
    run("src.run_revision","--workers","2","--seeds","42","7",
        "--output","results/revision_holdout",*args)
    run("scripts.revision_data_audit")
    run("src.revision_probes")
    run("src.revision_sensitivity")
    run("src.revision_scarcity")
    run("scripts.validate_revision")
    run("scripts.revision_label_metrics")
    run("scripts.revision_experiment_reports")
    run("scripts.make_all_tables_figures")
    run("scripts.make_all_tables_figures","--check")


if __name__=="__main__":
    main()
