"""主張（DAI/docs/主張_2026-11-12.md と 主張_採用した説明_2026-11-13.md）を導いた結果の控えを git に入れる形で作る．
- 結果の置き場（results/ は git に入らない）から，要約と記録の小さいファイルを gzip で写す（4 MB 以下と，名指しの大きいファイル）．
- 結果の置き場の全ファイルの sha256 と大きさの一覧を作る（丸ごとの控えと照合するため）．
使い方：.venv/bin/python DAI/evidence/make_evidence.py"""
import gzip, hashlib, json, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "主張_2026-11-12"
DIRS = ["dai1", "dai3", "dai11_probe", "dai14"]
SKIP_SUBDIRS = {"units", "rows", "rowsA", "pseudo", "eval/units", "source"}   # 単位ごとの大きい出力は sha256 の一覧だけ
FORCE = {"dai3/eval/decomp.csv", "dai1/source/summary.csv"}
LIMIT = 4 * 1024 * 1024


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    copied, listing = [], []
    for d in DIRS:
        base = ROOT / "results" / d
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT / "results").as_posix()
            listing.append((rel, p.stat().st_size, sha(p)))
            parts = p.relative_to(base).parts
            in_skip = any(parts[:len(s.split("/"))] == tuple(s.split("/")) for s in SKIP_SUBDIRS)
            if rel in FORCE or (not in_skip and p.stat().st_size <= LIMIT and p.suffix in (".csv", ".json", ".jsonl", ".done", ".txt")):
                dst = OUT / "results" / (rel + ".gz")
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "rb") as fi, open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as fo:
                    shutil.copyfileobj(fi, fo)
                copied.append(rel)
    with open(OUT / "sha256_results.tsv", "w") as f:
        f.write("path\tbytes\tsha256\n")
        for rel, n, h in listing:
            f.write(f"{rel}\t{n}\t{h}\n")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    json.dump(dict(made_at_commit=head, n_files=len(listing), total_bytes=sum(n for _, n, _ in listing), n_copied=len(copied),
                   copied=copied), open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)
    print(len(listing), "files listed,", len(copied), "copied")


if __name__ == "__main__":
    main()
