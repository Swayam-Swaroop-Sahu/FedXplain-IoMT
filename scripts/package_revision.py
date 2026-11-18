"""Assemble local review bundles after a successful LaTeX build."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def write_zip(target, files):
    temporary=target.with_name(target.stem+"-building.zip")
    with ZipFile(temporary,"w",ZIP_DEFLATED,compresslevel=3) as bundle:
        for source,archive_name in files:
            bundle.write(source,archive_name)
    with ZipFile(temporary) as bundle:
        bad=bundle.testzip()
        if bad:
            raise ValueError(f"Zip integrity check failed at {bad}")
    temporary.replace(target)
    print(target,target.stat().st_size)


def main():
    build=Path("overleaf_package")
    main_files=[Path(name) for name in ("access.tex","revision_tables.tex","revision_numbers.tex","refs/references.bib")]
    figures=[Path("figures/fig_attack_distribution.png"),Path("figures/fig_iomt_mlp_architecture.png"),
             *sorted(Path("figures").glob("revision_*.png"))]
    source_map=[(build/(p if p.parent.name=="refs" else p.name),p if p.parent.name=="refs" else p.name)
                for p in main_files+figures]
    for (packaged,_),source in zip(source_map,main_files+figures):
        assert packaged.read_bytes()==source.read_bytes(),f"Build source differs: {source}"
    assets=["access.pdf","access.bbl","ieeeaccess.cls","Logo.png","notaglineLogo.png","bullet.png"]
    write_zip(Path("overleaf_package.zip"),source_map+[(build/p,p) for p in assets])
    source=main_files+list(Path("figures").glob("*.png"))
    for folder,suffixes in {
        "src":{".py"},"scripts":{".py"},"tests":{".py"},"config":{".yml"},
        "docs":{".md",".csv",".json"},"refs":{".md",".json"},
        "results/revision":{".json",".md",".csv",".npz",".pth"},
        "results/revision_holdout":{".json",".md",".csv",".npz",".pth"},
        "results_v1_archive":{".json",".md"},
        "splits/revision":{".csv"},"splits/revision_holdout":{".csv"},
    }.items():
        source.extend(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix in suffixes)
    source+=list(map(Path,["README.md","requirements.lock","requirements.txt","requirements-dev.txt",
        "app.py","overleaf_package.zip","data/README.md","results/study_results.json",
        "results/generalization_results.json","results/cluster_fl_results.json"]))
    write_zip(Path("FedXplain-IoMT-revision.zip"),[(p,p.as_posix()) for p in sorted(set(source))])


if __name__=="__main__":
    main()
