"""
Экспорт кадров с дрейфом для модуля разметки/дообучения.

Кадры копятся при срабатывании порога; выдача — только ещё не отданные (pending).
"""
from __future__ import annotations

import json
import os
import threading
import time
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

_LOCK = threading.Lock()


def _ensure_dirs(base_dir: str) -> Dict[str, str]:
    frames_dir = os.path.join(base_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    return {
        "base": base_dir,
        "frames": frames_dir,
        "index": os.path.join(base_dir, "index.jsonl"),
        "sent": os.path.join(base_dir, "sent.json"),
    }


def _load_sent(sent_path: str) -> Set[str]:
    if not os.path.exists(sent_path):
        return set()
    try:
        with open(sent_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = data.get("sent_ids", [])
        return set(str(x) for x in ids)
    except Exception:
        return set()


def _save_sent(sent_path: str, sent_ids: Set[str]) -> None:
    tmp = sent_path + ".tmp"
    payload = {
        "sent_ids": sorted(sent_ids),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, sent_path)


def make_frame_id(job_id: str, frame_index: int) -> str:
    """Стабильный уникальный id кадра в рамках прогона."""
    return f"{job_id}_{int(frame_index):08d}"


def save_drift_frame(
    base_dir: str,
    *,
    job_id: str,
    frame_index: int,
    second: float,
    metric: str,
    value: float,
    threshold: float,
    image_bgr: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Сохраняет оригинал кадра (без оверлея) и дописывает запись в index.jsonl.
    Повтор с тем же frame_id не дублирует файл.
    """
    paths = _ensure_dirs(base_dir)
    frame_id = make_frame_id(job_id, frame_index)
    filename = f"{frame_id}.jpg"
    rel_path = os.path.join("frames", filename)
    abs_path = os.path.join(paths["frames"], filename)

    created_at = time.time()
    record = {
        "frame_id": frame_id,
        "job_id": job_id,
        "frame_index": int(frame_index),
        "second": float(second),
        "metric": str(metric),
        "value": float(value),
        "threshold": float(threshold),
        "path": rel_path.replace("\\", "/"),
        "created_at": created_at,
        "created_at_iso": datetime.utcfromtimestamp(created_at).isoformat() + "Z",
    }

    with _LOCK:
        if not os.path.exists(abs_path):
            ok = cv2.imwrite(abs_path, image_bgr)
            if not ok:
                return None
        # Если запись уже есть в index — не дублируем строку
        already = False
        if os.path.exists(paths["index"]):
            with open(paths["index"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if json.loads(line).get("frame_id") == frame_id:
                            already = True
                            break
                    except Exception:
                        continue
        if not already:
            with open(paths["index"], "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _iter_index(index_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(index_path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def list_pending(
    base_dir: str,
    *,
    job_id: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Кадры, ещё не помеченные как sent."""
    paths = _ensure_dirs(base_dir)
    with _LOCK:
        sent = _load_sent(paths["sent"])
        rows = _iter_index(paths["index"])

    pending: List[Dict[str, Any]] = []
    for r in rows:
        fid = str(r.get("frame_id", ""))
        if not fid or fid in sent:
            continue
        if job_id is not None and str(r.get("job_id")) != str(job_id):
            continue
        if since is not None:
            try:
                if float(r.get("created_at", 0)) < float(since):
                    continue
            except Exception:
                continue
        abs_path = os.path.join(paths["base"], str(r.get("path", "")).replace("/", os.sep))
        if not os.path.exists(abs_path):
            continue
        pending.append(r)
        if len(pending) >= max(1, int(limit)):
            break
    return pending


def mark_sent(base_dir: str, frame_ids: List[str]) -> int:
    """Помечает frame_id как отданные. Возвращает сколько новых добавлено в sent."""
    paths = _ensure_dirs(base_dir)
    ids = [str(x) for x in frame_ids if x]
    if not ids:
        return 0
    with _LOCK:
        sent = _load_sent(paths["sent"])
        before = len(sent)
        sent.update(ids)
        _save_sent(paths["sent"], sent)
        return len(sent) - before


def build_pending_zip(
    base_dir: str,
    zip_path: str,
    *,
    job_id: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = 500,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Собирает zip: frames/*.jpg + manifest.json.
    Возвращает (список записей в архиве, путь к zip).
    """
    pending = list_pending(base_dir, job_id=job_id, since=since, limit=limit)
    paths = _ensure_dirs(base_dir)
    manifest = {
        "count": len(pending),
        "job_id_filter": job_id,
        "since": since,
        "limit": limit,
        "created_at_iso": datetime.utcnow().isoformat() + "Z",
        "frames": pending,
    }
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for r in pending:
            abs_path = os.path.join(paths["base"], str(r.get("path", "")).replace("/", os.sep))
            arcname = f"frames/{os.path.basename(abs_path)}"
            zf.write(abs_path, arcname=arcname)
    return pending, zip_path
