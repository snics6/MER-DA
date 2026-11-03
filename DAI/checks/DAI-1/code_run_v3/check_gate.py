"""latest_report と判定の正規表現の確かめ（一時のフォルダの作り物のファイルで）。"""
import re, sys
from pathlib import Path
import dai1
tmp = Path(sys.argv[1])
for f in tmp.glob("*.md"): f.unlink()
for name in ["code.md", "code_v2.md", "code_v10.md", "code_v9b.md", "code_post.md", "code_v3_append.md", "codex_v20.md",
             "review_ml.md", "review_ml_v3.md", "review_ml_affect_v9.md"]:
    (tmp / name).write_text("x")
print("code の最新:", dai1.latest_report(tmp, "code"), "／ review_ml の最新:", dai1.latest_report(tmp, "review_ml"))
heads = {"判定：通す（記録 3 件）": "code", "判定：直してから通す": "code", "判定：**通す**": "code", "## 判定\n通す．": "code",
         "判定: 通す": "code", "判定：通さない": "code", "判定：通す／直してから通す／止める のどれか": "code",
         "判定：OK": "review_ml", "判定：条件つき OK": "review_ml", "判定：OK でない": "review_ml", "判定：OKAY": "review_ml",
         "前回の判定：通す": "code"}
for h, role in heads.items():
    ok = r"^判定[：:]\s*\**OK" if role.startswith("review_") else r"^判定[：:]\s*\**通す"
    print(repr(h), "→ 合格" if re.search(ok, h, re.M) else "→ 不合格")
