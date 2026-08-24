"""
JADD: чувствительностно-взвешенный сдвиг распределения признаков детектора.

Формула (как в research proposal):

    JADD² = || J · Δμ ||² + λ · d_B²(C_P, C_Q)

где
    Δμ = μ_Q − μ_P
    C_P = J · Σ_P · Jᵀ
    C_Q = J · Σ_Q · Jᵀ

    d_B²(C_P, C_Q) = tr( C_P + C_Q − 2 · (C_P^{1/2} · C_Q · C_P^{1/2})^{1/2} )

Итог: JADD = sqrt(JADD²).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.linalg import sqrtm


# Размерность признаков после проекции (стабильные Σ при умеренном N)
FEATURE_DIM = 256
# Регуляризация ковариаций / C (положительная определённость для sqrtm)
EPS_COV = 1e-6


def d_B_squared(C_P: np.ndarray, C_Q: np.ndarray) -> float:
    """
    Квадрат расстояния Бюреса между ковариациями C_P и C_Q:

        d_B²(C_P, C_Q) = tr( C_P + C_Q − 2 · (C_P^{1/2} · C_Q · C_P^{1/2})^{1/2} )
    """
    C_P = _symmetrize_psd(C_P)
    C_Q = _symmetrize_psd(C_Q)
    C_P_half = _matrix_sqrt(C_P)
    inner = _matrix_sqrt(C_P_half @ C_Q @ C_P_half)
    M = C_P + C_Q - 2.0 * inner
    return float(np.real(np.trace(M)))


def compute_jadd(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    Sigma_P: np.ndarray,
    Sigma_Q: np.ndarray,
    J: np.ndarray,
    lambda_: float,
) -> Dict[str, float]:
    """
    Считает JADD по обозначениям proposal.

    Args:
        mu_P, mu_Q: средние признаков (D,)
        Sigma_P, Sigma_Q: ковариации признаков (D, D)
        J: якобиан головы в эталонной точке, (m, D)
        lambda_: вклад члена формы (λ ≥ 0)

    Returns:
        dict с jadd, jadd_sq, mean_term, shape_term
    """
    delta_mu = mu_Q - mu_P  # Δμ = μ_Q − μ_P

    # C_P = J · Σ_P · Jᵀ ,  C_Q = J · Σ_Q · Jᵀ
    C_P = J @ Sigma_P @ J.T
    C_Q = J @ Sigma_Q @ J.T

    # || J · Δμ ||²
    mean_term = float(np.sum((J @ delta_mu) ** 2))

    # λ · d_B²(C_P, C_Q)
    d_b2 = d_B_squared(C_P, C_Q)
    shape_term = float(lambda_) * max(0.0, d_b2)

    jadd_sq = mean_term + shape_term
    jadd = float(np.sqrt(max(jadd_sq, 0.0)))

    return {
        "jadd": jadd,
        "jadd_sq": float(jadd_sq),
        "mean_term": mean_term,
        "shape_term": shape_term,
        "d_B_squared": float(max(0.0, d_b2)),
    }


def _symmetrize_psd(C: np.ndarray, eps: float = EPS_COV) -> np.ndarray:
    C = 0.5 * (C + C.T)
    C = C + eps * np.eye(C.shape[0], dtype=np.float64)
    return C.astype(np.float64)


def _matrix_sqrt(C: np.ndarray) -> np.ndarray:
    """Матричный квадратный корень (sqrtm), вещественная часть."""
    S = sqrtm(C)
    S = np.real(S)
    return 0.5 * (S + S.T)


def _estimate_mean_cov(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """μ и Σ по строкам матрицы признаков (N, D)."""
    if F.ndim != 2 or F.shape[0] == 0:
        raise ValueError("Нужна матрица признаков (N, D) с N≥1")
    mu = np.mean(F, axis=0)
    if F.shape[0] == 1:
        Sigma = np.eye(F.shape[1], dtype=np.float64) * EPS_COV
    else:
        Sigma = np.cov(F, rowvar=False)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]], dtype=np.float64)
        Sigma = _symmetrize_psd(np.atleast_2d(Sigma))
    return mu.astype(np.float64), Sigma.astype(np.float64)


def fit_head_jacobian(F_P: np.ndarray, G_P: np.ndarray) -> np.ndarray:
    """
    Локальная линейная карта головы g(f) ≈ J f + b по эталону P.

    J — якобиан размера m×D (как в proposal: ∂g/∂f в эталонной точке).
    Для линейной головы J постоянен и равен матрице весов.
    """
    N, D = F_P.shape
    m = G_P.shape[1]
    if N < 1:
        return np.zeros((m, D), dtype=np.float64)

    # G ≈ [F | 1] @ W_aug  →  W_aug: (D+1, m), J = W_aug[:-1].T
    F_aug = np.hstack([F_P, np.ones((N, 1), dtype=np.float64)])
    W_aug, _, _, _ = np.linalg.lstsq(F_aug, G_P, rcond=None)
    J = W_aug[:-1].T.astype(np.float64)  # (m, D)
    return J


def project_features(F: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """F (N, D_raw) → (N, D) через фиксированную проекцию proj (D_raw, D)."""
    return (F @ proj).astype(np.float64)


def make_random_projection(d_raw: int, d_out: int = FEATURE_DIM, seed: int = 42) -> np.ndarray:
    """Ортонормированная случайная проекция R^{d_raw} → R^{d_out}."""
    rng = np.random.default_rng(seed)
    d_out = min(d_out, d_raw)
    A = rng.normal(size=(d_raw, d_out))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q.astype(np.float64)


def prediction_vector(
    class_index: int,
    confidence: float,
    bbox: Sequence[float],
    image_wh: Tuple[int, int],
    num_classes: int,
) -> np.ndarray:
    """
    g(f): логиты/скоры классов (K) + 4 координаты бокса (норм. cx,cy,w,h).
    m = K + 4.
    """
    K = num_classes
    g = np.zeros(K + 4, dtype=np.float64)
    if 0 <= class_index < K:
        g[class_index] = float(confidence)
    w_img, h_img = image_wh
    x1, y1, x2, y2 = bbox
    w_img = max(float(w_img), 1.0)
    h_img = max(float(h_img), 1.0)
    cx = ((x1 + x2) * 0.5) / w_img
    cy = ((y1 + y2) * 0.5) / h_img
    bw = max(0.0, (x2 - x1) / w_img)
    bh = max(0.0, (y2 - y1) / h_img)
    g[K:] = (cx, cy, bw, bh)
    return g


class JADDCalculator:
    """
    Извлекает f (признаки до головы), строит J на эталоне P и считает JADD.

    Важно: НЕ вызываем YOLO.embed() на том же инстансе, что и detect —
    Ultralytics после embed() начинает возвращать Tensor вместо Results
    → ошибка "'Tensor' object has no attribute 'boxes'".
    Поэтому f берём из отдельного ResNet50 (тот же энкодер, что в drift_analyzer).
    """

    def __init__(
        self,
        yolo_model=None,
        class_names: Optional[List[str]] = None,
        lambda_: float = 1.0,
        feature_dim: int = FEATURE_DIM,
    ):
        # yolo_model намеренно не используем для embed (ломает detect на shared instance)
        self.yolo_model = None
        self.class_names = [c.lower() for c in (class_names or [])]
        self.lambda_ = float(lambda_)
        self.feature_dim = int(feature_dim)
        self._proj: Optional[np.ndarray] = None
        self._resnet = None
        self._feature_extractor = None

    def extract_f(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Вектор признаков f одного объекта (ResNet50 перед классификатором)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        feat = self._embed_resnet(crop_bgr)
        if feat is None:
            return None
        feat = np.asarray(feat, dtype=np.float64).ravel()
        nrm = np.linalg.norm(feat)
        if nrm > 1e-12:
            feat = feat / nrm
        return feat

    def _embed_resnet(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            from torchvision import models, transforms

            if self._feature_extractor is None:
                try:
                    net = models.resnet50(weights="IMAGENET1K_V2")
                except TypeError:
                    net = models.resnet50(pretrained=True)
                net.eval()
                self._feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
            tfm = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            x = tfm(rgb).unsqueeze(0)
            with torch.no_grad():
                feat = self._feature_extractor(x).flatten().numpy()
            return feat
        except Exception as e:
            print(f"JADD: ошибка ResNet-признаков: {e}")
            return None

    def detections_to_fg(
        self,
        detections: List[dict],
        image_shape: Tuple[int, int, int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Из детекций кадра → списки f и g.
        image_shape: (H, W, C)
        """
        h, w = image_shape[0], image_shape[1]
        K = max(1, len(self.class_names))
        name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        fs: List[np.ndarray] = []
        gs: List[np.ndarray] = []
        for det in detections:
            crop = det.get("crop")
            f = self.extract_f(crop)
            if f is None:
                continue
            cname = str(det.get("class_name", "")).lower()
            if self.class_names:
                ci = name_to_idx.get(cname, -1)
                if ci < 0:
                    # класс вне фильтра — пропускаем
                    continue
            else:
                ci = 0
            g = prediction_vector(
                class_index=ci if ci >= 0 else 0,
                confidence=float(det.get("confidence", 0.0)),
                bbox=det.get("bbox", [0, 0, 0, 0]),
                image_wh=(w, h),
                num_classes=K,
            )
            fs.append(f)
            gs.append(g)
        return fs, gs

    def _ensure_proj(self, d_raw: int) -> np.ndarray:
        if self._proj is None or self._proj.shape[0] != d_raw:
            self._proj = make_random_projection(d_raw, self.feature_dim)
        return self._proj

    def compute_from_windows(
        self,
        features_P: Sequence[np.ndarray],
        preds_P: Sequence[np.ndarray],
        features_Q: Sequence[np.ndarray],
        preds_Q: Sequence[np.ndarray],
    ) -> Optional[Dict[str, float]]:
        """
        Считает JADD между эталонным окном P и текущим Q.
        preds_Q не используются в формуле (нужны f и g на P для J; на Q — только распределение f).
        """
        if len(features_P) < 2 or len(features_Q) < 2:
            return None
        F_P_raw = np.stack([np.asarray(f, dtype=np.float64).ravel() for f in features_P], axis=0)
        F_Q_raw = np.stack([np.asarray(f, dtype=np.float64).ravel() for f in features_Q], axis=0)
        if F_P_raw.shape[1] != F_Q_raw.shape[1]:
            return None

        proj = self._ensure_proj(F_P_raw.shape[1])
        F_P = project_features(F_P_raw, proj)
        F_Q = project_features(F_Q_raw, proj)

        G_P = np.stack([np.asarray(g, dtype=np.float64).ravel() for g in preds_P], axis=0)
        if G_P.shape[0] != F_P.shape[0]:
            n = min(G_P.shape[0], F_P.shape[0])
            F_P, G_P = F_P[:n], G_P[:n]

        mu_P, Sigma_P = _estimate_mean_cov(F_P)
        mu_Q, Sigma_Q = _estimate_mean_cov(F_Q)

        # J — якобиан головы в эталонной точке (линейная аппроксимация по P)
        J = fit_head_jacobian(F_P, G_P)

        return compute_jadd(mu_P, mu_Q, Sigma_P, Sigma_Q, J, self.lambda_)

    def compute_stream(self, window_items: Sequence[Tuple[List[np.ndarray], List[np.ndarray]]]) -> Optional[Dict[str, float]]:
        """
        Скользящее окно как у analyze_drift_stream:
        старшая половина = P (эталон), младшая = Q (текущее).
        Каждый элемент окна: (list f, list g) по детекциям кадра.
        """
        if len(window_items) < 2:
            return None
        mid = len(window_items) // 2
        f_P, g_P = [], []
        f_Q, g_Q = [], []
        for fs, gs in window_items[:mid]:
            f_P.extend(fs)
            g_P.extend(gs)
        for fs, gs in window_items[mid:]:
            f_Q.extend(fs)
            g_Q.extend(gs)
        return self.compute_from_windows(f_P, g_P, f_Q, g_Q)
