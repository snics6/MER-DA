"""評価の回の守り frozen/eval_gate.json を，整理（2026-11-15）の後の置き場と sha256 で作り直す．

元の eval_gate.json（2026-11-05 の評価の回の前に作った．置き場は MER-DA の直下から）の鍵を，整理の対応表
docs/整理の対応表_2026-11-15.tsv で新しい置き場に置き換え，sha256 を今のファイルで計算し直す．
元の鍵と sha256 は _meta.before_reorg に残す．中身が元の sha256 と違うファイル（置き場の行を書き換えた台本と，
評価の回の後に 1 行直した docs/設計/実験1_設計.md）は _meta.changed_after_gate に並べる．
使い方：python tools/remap_eval_gate.py      （DAI の直下で）
"""
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "frozen" / "eval_gate.json"
MAP = ROOT / "docs" / "整理の対応表_2026-11-15.tsv"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = json.load(open(GATE, encoding="utf-8"))
    if "before_reorg" in old.get("_meta", {}):
        raise SystemExit("既に作り直してある")
    m = {}
    with open(MAP, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["new_path"]:
                m[r["old_path"]] = r["new_path"].removeprefix("DAI/")

    def new_key(k):
        if k in m:
            return m[k]
        if k.startswith("DAI/") and (ROOT / k[4:]).exists():          # 動かしていない DAI の中のファイル（frozen・run など）
            return k[4:]
        raise SystemExit(f"新しい置き場が分からない：{k}")

    req, changed = {}, []
    for k, h in old["required"].items():
        nk = new_key(k)
        nh = sha(ROOT / nk)
        req[nk] = nh
        if nh != h:
            changed.append({"path": nk, "old_path": k, "old_sha256": h, "new_sha256": nh})
    appr = {e: {role: new_key(p) for role, p in files.items()} for e, files in old["approvals"].items()}
    meta = dict(old["_meta"])
    meta["before_reorg"] = {"required": old["required"], "approvals": old["approvals"]}
    meta["changed_after_gate"] = changed
    meta["reorg_note"] = ("2026-11-15 の DAI の整理で置き場を変えた．鍵は DAI の直下からの相対パス．sha256 は整理の後のファイルで計算し直した．"
                          "台本の差分は置き場の行だけ（docs/整理の書き換えの記録_2026-11-15.tsv，checks/整理_2026-11-15/）．"
                          "docs/設計/実験1_設計.md は評価の回の後のコミット 48212d3 で 1 行直してあり，元の sha256 とは違う")
    out = {"_meta": meta, "required": req, "approvals": appr}
    GATE.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"required {len(req)} 本，中身が元と違う {len(changed)} 本：", ", ".join(c["path"] for c in changed))


if __name__ == "__main__":
    main()
