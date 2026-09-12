"""起動時に事前学習データを確認し、不足分だけ青空文庫から補充する。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://www.aozora.gr.jp"
INDEX_ZIP_URL = f"{BASE_URL}/index_pages/list_person_all_extended_utf8.zip"


def _download(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "DiscordExperienceAI/1.0 dataset updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _clean_text(raw: bytes) -> str:
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 青空文庫の本文以外の説明部分を落とす
    if "---------------------------------------" in text:
        text = text.split("---------------------------------------", 1)[1]
    if "底本：" in text:
        text = text.split("底本：", 1)[0]
    if "底本：" not in text and "［＃底本" in text:
        text = text.split("［＃底本", 1)[0]

    # 青空文庫注記を学習本文から除去。ルビ本文は残す。
    text = re.sub(r"｜([^《]+)《[^》]+》", r"\1", text)
    text = re.sub(r"《[^》]+》", "", text)
    text = re.sub(r"［＃[^］]*］", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_catalog(raw_zip: bytes):
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            text = f.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _work_key(row: dict) -> str:
    return str(row.get("作品ID", "")).strip()


def update_dataset(config: dict) -> dict:
    dc = config.get("dataset", {})
    root = Path("data/aozora")
    root.mkdir(parents=True, exist_ok=True)
    corpus = Path("data/pretrain.txt")
    manifest_path = root / "manifest.json"
    catalog_path = root / "list_person_all_extended_utf8.zip"

    min_chars = int(dc.get("min_corpus_chars", 5_000_000))
    max_new = int(dc.get("max_new_works_per_start", 20))
    timeout = int(dc.get("download_timeout_seconds", 30))
    enabled = bool(dc.get("auto_fill", True))
    if not enabled:
        return {"added": 0, "chars": len(corpus.read_text(encoding="utf-8")) if corpus.exists() else 0, "skipped": True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"works": {}}
    works = manifest.setdefault("works", {})

    # 起動ごとに公式カタログを確認する。既存作品は本文を再取得しない。
    try:
        catalog = _download(INDEX_ZIP_URL, timeout)
        catalog_path.write_bytes(catalog)
    except Exception as exc:
        print(f"[dataset] 青空文庫カタログ更新失敗: {exc}")
        catalog = catalog_path.read_bytes() if catalog_path.exists() else None
        if catalog is None:
            return {"added": 0, "chars": 0, "error": str(exc)}

    rows = _read_catalog(catalog)
    candidates = {}
    for row in rows:
        wid = _work_key(row)
        url = (row.get("テキストファイルURL") or "").strip()
        if not wid or not url or not url.startswith(BASE_URL):
            continue
        if row.get("作品著作権フラグ") != "なし" or row.get("人物著作権フラグ") != "なし":
            continue
        candidates[wid] = row

    current_chars = len(corpus.read_text(encoding="utf-8")) if corpus.exists() else 0
    added = 0
    failures = 0
    # 新しい更新日を優先。ただし毎回同じ順序なので再現可能。
    pending = [r for wid, r in candidates.items() if wid not in works]
    pending.sort(key=lambda r: (r.get("最終更新日", ""), r.get("作品ID", "")), reverse=True)

    with corpus.open("a", encoding="utf-8") as out:
        for row in pending:
            if current_chars >= min_chars or added >= max_new:
                break
            wid = _work_key(row)
            try:
                blob = _download(row["テキストファイルURL"], timeout)
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    txt_name = next(n for n in z.namelist() if not n.endswith("/"))
                    text = _clean_text(z.read(txt_name))
                if len(text) < 100:
                    raise ValueError("本文が短すぎます")
                out.write(f"\n\n<|document|>\n{text}\n<|end_document|>\n")
                current_chars += len(text)
                works[wid] = {
                    "title": row.get("作品名", ""),
                    "updated": row.get("最終更新日", ""),
                    "url": row.get("テキストファイルURL", ""),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "chars": len(text),
                }
                added += 1
                print(f"[dataset] 追加: {row.get('作品名', wid)} ({len(text):,} chars)")
            except Exception as exc:
                failures += 1
                print(f"[dataset] 取得失敗 {wid}: {exc}")
            time.sleep(float(dc.get("request_interval_seconds", 0.2)))

    manifest["catalog_url"] = INDEX_ZIP_URL
    manifest["last_check"] = time.time()
    manifest["target_chars"] = min_chars
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"added": added, "chars": current_chars, "failures": failures, "target": min_chars}
