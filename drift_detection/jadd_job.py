"""
Единый job основного роута дрейфа: video / archive / rtsp.
"""
from __future__ import annotations

import os
import shutil
import time
from collections import deque, defaultdict
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from jadd import JADDCalculator
from drift_frames_export import save_drift_frame
from drift_schedule import is_schedule_active


def run_jadd_job(
    *,
    job_id: str,
    source_type: str,
    source_path: str,
    object_classes: List[str],
    frame_stride: int,
    drift_window_sec: float,
    drift_window_photos: int,
    only_frames_with_detections: bool,
    loop_video: bool,
    loop_count: int,
    boundaries: Optional[Any],
    lambda_: float,
    ema_alpha: Optional[float],  # EMA отключено; параметр оставлен для совместимости
    alert_metric: str,
    alert_threshold: float,
    schedule: Optional[Dict[str, Any]],
    # колбэки/зависимости из api.py (чтобы не плодить циклы импорта)
    processed_frames_dir: str,
    drift_export_dir: str,
    drift_frame_max_edge: int,
    init_video_job: Callable,
    update_video_job: Callable,
    get_pretrained_detector: Callable,
    convert_numpy_types: Callable,
    draw_detections: Callable,
    get_or_assign_class_colors: Callable,
    record_drift_metrics: Callable,
    normalize_drift_components: Callable,
    compute_agg_base: Callable,
    default_component_flags: Callable,
    get_segment_index_for_video_second: Callable,
    extract_archive_to_temp_dir: Callable,
    is_stop_requested: Callable[[str], bool],
    video_jobs_lock,
    video_jobs: Dict[str, Any],
    jadd_gauge,
    aggregate_v5_ema_gauge=None,  # EMA отключено
) -> None:
    source_type = (source_type or "video").strip().lower()
    if source_type not in ("video", "archive", "rtsp"):
        update_video_job(job_id, status="error", error=f"Неизвестный source_type: {source_type}")
        return

    output_dir = os.path.join(processed_frames_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)
    init_video_job(job_id, output_dir)
    update_video_job(job_id, source_type=source_type)

    alert_metric = (alert_metric or "v5").strip().lower()
    # if alert_metric not in ("v5", "v5_ema", "jadd"):
    if alert_metric not in ("v5", "jadd"):
        alert_metric = "v5"
    component_flags = default_component_flags()

    det = get_pretrained_detector(object_classes)
    detector, analyzer = det.detector, det.analyzer
    jadd_calc = JADDCalculator(
        yolo_model=None,
        class_names=object_classes,
        lambda_=lambda_,
    )
    class_colors: Dict[str, tuple] = {}

    # EMA отключено
    # _ema_alpha = None
    # try:
    #     if ema_alpha is not None and float(ema_alpha) > 0.0 and float(ema_alpha) <= 1.0:
    #         _ema_alpha = float(ema_alpha)
    # except Exception:
    #     _ema_alpha = None
    # v5_ema_state: Optional[float] = None
    _ = (ema_alpha, aggregate_v5_ema_gauge)
    schedule_was_active: Optional[bool] = None

    processed_frames = 0
    start_time = time.time()
    last_detection_second = None

    if source_type == "archive":
        frame_window = deque(maxlen=max(2, int(drift_window_photos)))
        feat_window = deque(maxlen=max(2, int(drift_window_photos)))
    else:
        # размер окна для video/rtsp уточним после открытия стрима (нужен fps)
        frame_window = deque(maxlen=2)
        feat_window = deque(maxlen=2)

    def _handle_frame(frame, frame_index: int, second: float, seg_idx: Optional[int]) -> bool:
        """Обработать один кадр. False = остановить job."""
        # nonlocal processed_frames, last_detection_second, v5_ema_state, schedule_was_active
        nonlocal processed_frames, last_detection_second, schedule_was_active

        if is_stop_requested(job_id):
            return False

        schedule_active = is_schedule_active(schedule)
        if schedule_was_active is True and not schedule_active:
            frame_window.clear()
            feat_window.clear()
            # v5_ema_state = None  # EMA отключено
        schedule_was_active = schedule_active

        if not schedule_active:
            processed_frames += 1
            with video_jobs_lock:
                job = video_jobs.get(job_id)
                if job:
                    job["processed_frames"] = processed_frames
                    job["last_metrics"] = {
                        "frame_index": frame_index,
                        "second": second,
                        "processed_frames": processed_frames,
                        "detections_count": 0,
                        "drift_metrics": {},
                        "schedule_active": False,
                        "distortion_stage": "schedule_skip",
                    }
            return True

        processed_frame = frame
        h, w = processed_frame.shape[:2]
        if max(h, w) > drift_frame_max_edge:
            scale = drift_frame_max_edge / max(h, w)
            small_frame = cv2.resize(
                processed_frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small_frame = processed_frame.copy()

        detections = detector.detect_objects(processed_frame)
        if not only_frames_with_detections or len(detections) > 0:
            frame_window.append(small_frame)
            fs, gs = jadd_calc.detections_to_fg(detections, processed_frame.shape)
            feat_window.append((fs, gs))

        drift_metrics_raw = None
        if len(frame_window) >= 2:
            try:
                drift_metrics_raw = analyzer.analyze_drift_stream(list(frame_window))
            except Exception as e:
                print(f"Ошибка расчёта дрейфа (stream): {e}")
        drift_metrics = convert_numpy_types(drift_metrics_raw or {})

        jadd_metrics = None
        if len(feat_window) >= 2:
            try:
                jadd_metrics = jadd_calc.compute_stream(list(feat_window))
            except Exception as e:
                print(f"Ошибка расчёта JADD: {e}")
        if jadd_metrics:
            drift_metrics["jadd"] = jadd_metrics.get("jadd")
            drift_metrics["jadd_sq"] = jadd_metrics.get("jadd_sq")
            drift_metrics["jadd_mean_term"] = jadd_metrics.get("mean_term")
            drift_metrics["jadd_shape_term"] = jadd_metrics.get("shape_term")
            try:
                jadd_gauge.set(float(jadd_metrics["jadd"]))
            except Exception:
                pass

        if detections:
            last_detection_second = second
        get_or_assign_class_colors(detections, class_colors)
        overlay = draw_detections(processed_frame, detections, class_colors)
        cv2.imwrite(os.path.join(output_dir, f"frame_{processed_frames:06d}.jpg"), overlay)
        processed_frames += 1

        metrics_entry = {
            "frame_index": frame_index,
            "second": second,
            "processed_frames": processed_frames,
            "detections_count": len(detections),
            "drift_metrics": drift_metrics,
            "schedule_active": True,
            "distortion_stage": "original",
        }
        with video_jobs_lock:
            job = video_jobs.get(job_id)
            if job:
                job["metrics_history"].append(metrics_entry)
                job["last_metrics"] = metrics_entry
                job["processed_frames"] = processed_frames
                job["last_detection_second"] = last_detection_second

        if drift_metrics:
            metrics_payload = drift_metrics.copy()
            metrics_payload["total_detections"] = len(detections)
            counts = {c: 0 for c in object_classes}
            conf_by_class = {}
            for c in object_classes:
                confs = [
                    float(d.get("confidence", 0.0))
                    for d in detections
                    if str(d.get("class_name", "")).lower() == c
                ]
                if confs:
                    conf_by_class[c] = sum(confs) / len(confs)
            if not conf_by_class and detections:
                conf_by_class = {
                    "_all": float(np.mean([float(d.get("confidence", 0.0)) for d in detections]))
                }
            for d in detections:
                cn = str(d.get("class_name", "object")).lower()
                if cn in counts:
                    counts[cn] += 1

            record_drift_metrics(
                metrics_payload,
                time.time() - start_time,
                job_id=job_id,
                video_second=second,
                detections_by_class=counts,
                confidence_by_class=conf_by_class if conf_by_class else None,
                segment_index=seg_idx,
                component_flags=component_flags,
                drift_window_sec=drift_window_sec if source_type != "archive" else float(drift_window_photos),
            )

            v5_raw = None
            if conf_by_class:
                try:
                    norm = normalize_drift_components(metrics_payload)
                    if norm is not None:
                        agg = compute_agg_base(norm, component_flags)
                        mean_conf = sum(conf_by_class.values()) / len(conf_by_class)
                        v5_raw = min(1.0, float(agg) / (0.5 + 0.5 * float(mean_conf)))
                        # EMA отключено
                        # if _ema_alpha is not None:
                        #     if v5_ema_state is None:
                        #         v5_ema_state = float(v5_raw)
                        #     else:
                        #         v5_ema_state = _ema_alpha * float(v5_raw) + (1.0 - _ema_alpha) * float(v5_ema_state)
                        #     try:
                        #         if aggregate_v5_ema_gauge is not None:
                        #             aggregate_v5_ema_gauge.set(float(v5_ema_state))
                        #     except Exception:
                        #         pass
                except Exception:
                    pass

            alert_value = None
            if alert_metric == "jadd":
                alert_value = drift_metrics.get("jadd")
            # elif alert_metric == "v5_ema":
            #     alert_value = v5_ema_state
            else:
                alert_value = v5_raw
            if alert_value is not None:
                try:
                    if float(alert_value) >= float(alert_threshold):
                        save_drift_frame(
                            drift_export_dir,
                            job_id=job_id,
                            frame_index=frame_index,
                            second=second,
                            metric=alert_metric,
                            value=float(alert_value),
                            threshold=float(alert_threshold),
                            image_bgr=processed_frame,
                        )
                except Exception as e:
                    print(f"Ошибка сохранения drift-кадра: {e}")
        return True

    extract_dir = None
    try:
        if source_type == "archive":
            try:
                extract_dir, image_paths = extract_archive_to_temp_dir(source_path)
            except Exception as e:
                update_video_job(job_id, status="error", error=f"Ошибка распаковки архива: {e}")
                return
            if not image_paths:
                update_video_job(job_id, status="error", error="В архиве не найдено изображений (jpg/png)")
                return
            update_video_job(job_id, total_frames=len(image_paths))

            archive_boundaries = boundaries if isinstance(boundaries, list) else None
            segment_index = 0
            for photo_index, img_path in enumerate(image_paths):
                if is_stop_requested(job_id):
                    update_video_job(job_id, status="stopped", message="Остановлено пользователем", finished_at=time.time())
                    return
                current_photo_name = os.path.splitext(os.path.basename(img_path))[0].lower()
                if archive_boundaries and archive_boundaries and isinstance(archive_boundaries[0], str):
                    while (
                        segment_index + 1 < len(archive_boundaries)
                        and archive_boundaries[segment_index + 1] == current_photo_name
                    ):
                        segment_index += 1
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                seg_idx = (
                    segment_index
                    if archive_boundaries and isinstance(archive_boundaries[0], str) and segment_index < len(archive_boundaries)
                    else None
                )
                if not _handle_frame(frame, photo_index, float(photo_index), seg_idx):
                    update_video_job(job_id, status="stopped", message="Остановлено пользователем", finished_at=time.time())
                    return
        else:
            # video или rtsp
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                update_video_job(
                    job_id,
                    status="error",
                    error=f"Не удалось открыть источник ({source_type}): {source_path}",
                )
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 1e-3:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if source_type == "rtsp":
                total_frames = 0
            update_video_job(job_id, total_frames=total_frames if total_frames > 0 else None)

            frames_per_window = max(2, int(float(drift_window_sec) * fps / max(1, int(frame_stride))))
            frame_window = deque(maxlen=frames_per_window)
            feat_window = deque(maxlen=frames_per_window)

            global_frame_index = 0
            loops_done = 0
            # для rtsp loop не используем — крутим до stop / обрыва
            use_loop = bool(loop_video) and source_type == "video"

            while True:
                if is_stop_requested(job_id):
                    cap.release()
                    update_video_job(job_id, status="stopped", message="Остановлено пользователем", finished_at=time.time())
                    return

                ret, frame = cap.read()
                if not ret:
                    if source_type == "rtsp":
                        # краткий ретрай при обрыве RTSP
                        time.sleep(1.0)
                        cap.release()
                        if is_stop_requested(job_id):
                            update_video_job(job_id, status="stopped", message="Остановлено пользователем", finished_at=time.time())
                            return
                        cap = cv2.VideoCapture(source_path)
                        if not cap.isOpened():
                            update_video_job(job_id, status="error", error="RTSP поток недоступен", finished_at=time.time())
                            return
                        continue
                    loops_done += 1
                    if not use_loop or (loop_count > 0 and loops_done >= loop_count):
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                global_frame_index += 1
                current_second = global_frame_index / fps
                if frame_stride > 1 and (global_frame_index % frame_stride != 0):
                    continue

                video_boundaries = boundaries if boundaries and isinstance(boundaries, list) and boundaries and isinstance(boundaries[0], (int, float)) else None
                seg_idx = (
                    get_segment_index_for_video_second(video_boundaries, current_second)
                    if video_boundaries
                    else None
                )
                if not _handle_frame(frame, global_frame_index, current_second, seg_idx):
                    cap.release()
                    update_video_job(job_id, status="stopped", message="Остановлено пользователем", finished_at=time.time())
                    return

            cap.release()

        update_video_job(
            job_id,
            status="completed",
            message=f"Обработка ({source_type}) завершена",
            finished_at=time.time(),
        )
    except Exception as e:
        update_video_job(job_id, status="error", error=str(e), finished_at=time.time())
    finally:
        if extract_dir:
            try:
                shutil.rmtree(extract_dir, ignore_errors=True)
            except Exception:
                pass
        if source_type in ("video", "archive") and source_path and os.path.exists(source_path):
            try:
                os.unlink(source_path)
            except Exception:
                pass
