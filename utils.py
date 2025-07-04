# app/services/audio.py
from __future__ import annotations
import asyncio, logging, pathlib, uuid
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any
import tempfile, os, shutil
import assemblyai as aai
from pydub import AudioSegment
from speechbrain.inference import SpeakerRecognition

from config import settings

logger = logging.getLogger(__name__)

# ───────────────────────────── temp-file helper ─────────────────────────────
@contextmanager
def temp_wav(raw: bytes, suffix: str = ".wav"):
    tmp_dir = pathlib.Path(settings.temp_dir or "tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / f"{uuid.uuid4()}{suffix}"
    path.write_bytes(raw)
    try:
        yield str(path)
    finally:
        path.unlink(missing_ok=True)

# ───────────────────── AssemblyAI transcription (async) ─────────────────────
async def transcribe(path: str, speakers_expected: int, api_key: str):
    aai.settings.api_key = api_key
    cfg = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=speakers_expected,
    )
    tr = aai.Transcriber()
    loop = asyncio.get_running_loop()
    logger.info("Transcribing %s", path)
    return await loop.run_in_executor(None, tr.transcribe, path, cfg)

# ─────────────────────── cached SpeechBrain ECAPA model ─────────────────────
_model: SpeakerRecognition | None = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading SpeechBrain ECAPA model…")
        _model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=".cache/spk_model",
        )
    return _model

# ─────────────────────────── segmentation helpers ───────────────────────────
def segment_by_speaker(wav: str, transcript):
    audio = AudioSegment.from_wav(wav)
    combined: Dict[int, AudioSegment] = {}
    rows: List[List[Any]] = []
    for u in transcript.utterances:
        start_ms, end_ms = int(u.start), int(u.end)
        seg = audio[start_ms:end_ms]
        combined.setdefault(u.speaker, AudioSegment.empty())
        combined[u.speaker] += seg
        rows.append([u.speaker, start_ms/1000, end_ms/1000, u.text])
    return combined, rows

def export_segments(combined: Dict[int, AudioSegment]) -> Dict[int, str]:
    """Export clips into a *dedicated* temp dir and return {spk: path}.

    The directory is created with :pyfunc:`tempfile.mkdtemp` so it can be
    removed in one shot after speaker-matching.
    """
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="clips_"))
    paths: Dict[int, str] = {}
    for spk, seg in combined.items():
        if len(seg) < 100:
            logger.warning("Skip speaker %s - clip too short (%d ms)", spk, len(seg))
            continue
        fp = tmp_dir / f"speaker_{spk}.wav"
        seg.export(fp, format="wav")
        paths[spk] = str(fp)
    return paths, tmp_dir 

# ───────────────────── similarity & speaker mapping ─────────────────────────
def similarity(model: SpeakerRecognition, file1: str, file2: str) -> float:
    """Return cosine similarity; if either WAV is empty, return 0.0."""
    import soundfile as sf
    for fp in (file1, file2):
        if sf.info(fp).frames == 0:
            logger.warning("Empty audio %s – similarity forced to 0", fp)
            return 0.0
    emb1 = model.encode_batch(model.load_audio(file1))
    emb2 = model.encode_batch(model.load_audio(file2))
    return model.similarity(emb1, emb2).item()

def identify(model, clips, refs, df, threshold: float):
    """Label DataFrame speakers using reference clips (or generic fallback)."""
    if not any(refs.values()):
        df = df.copy()
        df["speaker"] = df["speaker"].apply(lambda s: f"Speaker {s}")
        return df

    mapping: Dict[int, str] = {}
    for spk, path in clips.items():
        best, label = 0.0, "Unknown"
        for name, ref in refs.items():
            if not ref:
                continue
            score = similarity(model, path, ref)
            if score > best:
                best, label = score, name
        mapping[spk] = label if best >= threshold else "Unknown"

    df["speaker"] = df["speaker"].map(mapping).fillna("Unknown")
    return df
