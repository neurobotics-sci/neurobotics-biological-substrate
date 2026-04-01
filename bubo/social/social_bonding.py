"""
bubo/social/social_bonding.py — Bubo v50.0
Social bonding system: face recognition → oxytocin → amygdala modulation.

BIOLOGY:
  The primate social brain is a tightly coupled system:
  - Fusiform Face Area (FFA): fast, expert face recognition (~120ms)
  - Oxytocin (OXT): paraventricular nucleus → posterior pituitary → blood
    Half-life: ~3 minutes in plasma, but central effects longer (30-60min)
    Actions: reduces amygdala reactivity to social threats (LeDoux + Adolphs)
             increases approach motivation (NAc DA release)
             strengthens social memory (hippocampal OXT receptors)
  - Social reward: VS (ventral striatum) codes RPE for social interactions
  - Right hemisphere specialisation: face recognition is right-lateralised
    (prosopagnosia from right fusiform lesions)

IMPLEMENTATION:
  Face detection:   lightweight cascade or YOLO-tiny (< 2ms on Orin GPU)
  Face recognition: MobileNetV3-Small + ArcFace embedding (128-dim, < 5ms)
  Bond database:    sqlite3 table (face_id TEXT, embedding BLOB, bond REAL, n_encounters INT)
  Bond update:      Hebbian social learning — each positive interaction → bond += Δ

BOND LEVELS AND EFFECTS:
  Bond 0.0: stranger  → threat_weight 1.0, DA ±0.0, eye_contact 0.0
  Bond 0.3: acquaint  → threat_weight 0.75, DA +0.10, eye_contact 0.3
  Bond 0.7: friend    → threat_weight 0.40, DA +0.25, sero +0.15, eye_contact 0.7
  Bond 1.0: bonded    → threat_weight 0.15, DA +0.40, sero +0.25, eye_contact 1.0

OXYTOCIN MODEL:
  dOXT/dt = k_on × social_contact × bond_level - k_off × OXT
  k_on  = 0.05/s (release rate per positive contact)
  k_off = 0.003/s (τ ≈ 330s ≈ 5.5 min clearance)
  OXT → amygdala_threat_scale = exp(-2 × OXT)  (exponential suppression)
  OXT → DA_boost = 0.15 × OXT  (social reward)
"""
import time, json, logging, sqlite3, numpy as np
from pathlib import Path
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("SocialBonding")
DB_PATH = Path("/opt/bubo/data/bubo_faces.db")

BOND_EFFECTS = [
    # (min_bond, max_bond, threat_weight, da_boost, sero_boost, label)
    (0.0, 0.15, 1.00,  0.00, 0.00, "stranger"),
    (0.15,0.40, 0.80,  0.05, 0.05, "acquaintance"),
    (0.40,0.65, 0.55,  0.15, 0.10, "familiar"),
    (0.65,0.85, 0.35,  0.28, 0.18, "friend"),
    (0.85,1.01, 0.15,  0.42, 0.28, "bonded"),
]

def bond_effects(bond_level: float) -> dict:
    for mn, mx, tw, da, sero, label in BOND_EFFECTS:
        if mn <= bond_level < mx:
            return {"threat_weight": tw, "da_boost": da, "sero_boost": sero, "label": label}
    return {"threat_weight": 1.0, "da_boost": 0.0, "sero_boost": 0.0, "label": "stranger"}


class FaceDatabase:
    """SQLite store for known face embeddings and bond levels."""

    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS faces (
            face_id TEXT PRIMARY KEY, name TEXT,
            embedding BLOB NOT NULL,
            bond_level REAL DEFAULT 0.0,
            n_encounters INTEGER DEFAULT 0,
            last_seen_ns INTEGER DEFAULT 0)""")
        self._conn.commit()

    def find_closest(self, embedding: np.ndarray, threshold: float = 0.5) -> dict:
        """Cosine similarity search. Returns best match or stranger."""
        rows = self._conn.execute(
            "SELECT face_id, name, embedding, bond_level FROM faces").fetchall()
        if not rows:
            return {"face_id": None, "name": "stranger", "bond_level": 0.0, "similarity": 0.0}
        q = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_sim = -1.0; best_row = None
        for row in rows:
            e = np.frombuffer(row[2], dtype=np.float32)
            if len(e) != 128: continue
            sim = float(np.dot(e / (np.linalg.norm(e) + 1e-8), q))
            if sim > best_sim:
                best_sim = sim; best_row = row
        if best_sim < threshold or best_row is None:
            return {"face_id": None, "name": "stranger", "bond_level": 0.0, "similarity": best_sim}
        self._conn.execute(
            "UPDATE faces SET n_encounters=n_encounters+1, last_seen_ns=? WHERE face_id=?",
            (time.time_ns(), best_row[0]))
        self._conn.commit()
        return {"face_id": best_row[0], "name": best_row[1],
                "bond_level": float(best_row[3]), "similarity": best_sim}

    def update_bond(self, face_id: str, delta: float):
        """Increase bond level by delta (Hebbian social learning)."""
        if face_id is None: return
        self._conn.execute(
            "UPDATE faces SET bond_level=MIN(1.0, bond_level+?) WHERE face_id=?",
            (delta, face_id))
        self._conn.commit()

    def register_face(self, name: str, embedding: np.ndarray, initial_bond: float = 0.0):
        """Register a new known face."""
        import uuid
        face_id = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT OR REPLACE INTO faces (face_id, name, embedding, bond_level) VALUES(?,?,?,?)",
            (face_id, name, embedding.astype(np.float32).tobytes(), initial_bond))
        self._conn.commit()
        return face_id

    def n_known(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]


class MobileNetFaceEmbedder:
    """
    Lightweight face embedding using MobileNetV3-Small + ArcFace.
    On Jetson Orin 8GB (sm_87 INT8): ~5ms per face.
    Falls back to random embedding (simulation mode) without GPU.
    """

    def __init__(self):
        self._model = None
        self._sim_mode = True
        try:
            import torch, torchvision
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # In production: load pre-trained ArcFace model
            # Here: use MobileNetV3 feature extractor as embedding backbone
            model = torchvision.models.mobilenet_v3_small(pretrained=False)
            model.classifier = torch.nn.Linear(576, 128)  # ArcFace head
            model.eval()
            self._model = model.to(device)
            self._device = device
            self._sim_mode = False
            logger.info(f"Face embedder on {device} (128-dim ArcFace)")
        except Exception as e:
            logger.warning(f"Face embedder fallback to simulation: {e}")

    def embed(self, face_crop_array: np.ndarray) -> np.ndarray:
        """
        face_crop_array: (112, 112, 3) uint8 normalised face crop
        Returns: (128,) float32 L2-normalised embedding
        """
        if self._sim_mode or face_crop_array is None:
            # Simulation: deterministic embedding from image statistics
            flat = face_crop_array.flatten() if face_crop_array is not None else np.zeros(112*112*3)
            rng  = np.random.default_rng(int(np.sum(flat[:32])) % (2**31))
            emb  = rng.standard_normal(128).astype(np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)
        try:
            import torch
            x = torch.from_numpy(face_crop_array.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
            x = x.to(self._device)
            with torch.no_grad():
                emb = self._model(x).cpu().numpy()[0]
            return (emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32)
        except Exception:
            return np.zeros(128, dtype=np.float32)


class OxytocinSystem:
    """
    Simplified oxytocin dynamics.
    Peripheral OXT half-life: ~3min; central effects: ~30-60min.
    We model central OXT as the relevant variable.
    """
    K_ON  = 0.05    # release rate (/s per unit social contact)
    K_OFF = 0.003   # clearance rate (/s → τ ≈ 330s)

    def __init__(self):
        self._level = 0.0; self._t = time.time()

    def step(self, social_contact: float, bond_level: float) -> float:
        """social_contact: 0=no contact, 1=full positive interaction"""
        dt = max(time.time() - self._t, 0.001); self._t = time.time()
        release = self.K_ON * social_contact * bond_level
        self._level = float(np.clip(
            self._level * np.exp(-self.K_OFF * dt) + release * dt, 0, 1))
        return self._level

    @property
    def amygdala_suppression(self) -> float:
        """OXT suppresses amygdala threat response: 1.0 at OXT=0, → 0.15 at OXT=1"""
        return float(np.exp(-2.0 * self._level))

    @property
    def da_boost(self) -> float:
        return float(0.15 * self._level)

    @property
    def level(self) -> float:
        return self._level


class SocialBondingSystem:
    """
    Full social bonding pipeline.
    Receives face detections from visual node, computes bond effects,
    broadcasts modulation signals to amygdala and hypothalamus.
    """
    HZ = 10   # face recognition rate

    def __init__(self, bus: NeuralBus):
        self._bus    = bus
        self._db     = FaceDatabase()
        self._embed  = MobileNetFaceEmbedder()
        self._oxt    = OxytocinSystem()
        self._bond_history: deque = deque(maxlen=50)
        self._current_face_id   = None
        self._current_bond      = 0.0
        self._current_name      = "stranger"
        self._gaze_maintained   = False
        self._t_gaze_start      = 0.0
        self._lock              = threading.Lock()
        logger.info(f"SocialBondingSystem | {self._db.n_known()} known faces")

    def process_face(self, face_crop: np.ndarray, bbox: list) -> dict:
        """
        Full pipeline: face detected → embed → match → bond effects.
        Returns recognition result with bond modulation signals.
        """
        emb    = self._embed.embed(face_crop)
        match  = self._db.find_closest(emb, threshold=0.55)
        effects = bond_effects(match["bond_level"])

        # Update oxytocin
        social_contact = 1.0 if match["similarity"] > 0.7 else 0.3
        oxt_level = self._oxt.step(social_contact, match["bond_level"])

        # Amygdala threat weight: bond × oxytocin suppression
        threat_weight = float(effects["threat_weight"] * self._oxt.amygdala_suppression)

        # Gaze direction toward recognised face
        cx_norm = (bbox[0] + bbox[2]) / 2 / 640.0 - 0.5  # normalised x centre
        cy_norm = (bbox[1] + bbox[3]) / 2 / 480.0 - 0.5

        with self._lock:
            self._current_face_id = match["face_id"]
            self._current_bond    = match["bond_level"]
            self._current_name    = match["name"]

        result = {
            "face_id":       match["face_id"],
            "name":          match["name"],
            "bond_level":    match["bond_level"],
            "similarity":    match["similarity"],
            "label":         effects["label"],
            "threat_weight": threat_weight,
            "da_boost":      effects["da_boost"] + self._oxt.da_boost,
            "sero_boost":    effects["sero_boost"],
            "oxt_level":     oxt_level,
            "bbox":          bbox,
            "gaze_target":   [float(cx_norm * 45), float(-cy_norm * 35)],  # degrees
            "timestamp_ns":  time.time_ns(),
        }
        return result

    def positive_interaction_update(self):
        """Call when a positive social interaction completes (approach + gaze)."""
        with self._lock:
            fid = self._current_face_id; bond = self._current_bond
        if fid is not None:
            delta = 0.02 * (1.0 - bond)  # diminishing returns
            self._db.update_bond(fid, delta)

    def get_current_state(self) -> dict:
        with self._lock:
            return {"face_id": self._current_face_id,
                    "name": self._current_name,
                    "bond_level": self._current_bond,
                    "oxt_level": self._oxt.level}
