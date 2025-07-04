from __future__ import annotations
import asyncio, io, logging, pandas as pd
import streamlit as st
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder
from typing import Dict
import shutil, os, tempfile, pathlib
import utils as svc

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Speaker Diarization & Identification")

# ─── UI helper ─────────────────────────────────────────────────────────────
def audio_input(label: str, key: str):
    st.subheader(label)
    upload_tab, record_tab = st.tabs(["Upload", "Record"])

    with upload_tab:
        fu = st.file_uploader(f"Upload {label} (WAV)", ["wav"], key=f"up_{key}")
        if fu:
            return fu.getvalue(), fu.name

    with record_tab:
        raw = audio_recorder(key=f"rec_{key}")
        if raw:
            st.audio(raw, format="audio/wav")
            dur = len(AudioSegment.from_file(io.BytesIO(raw), format="wav")) / 1000
            st.caption(f"Duration: {dur:.1f}s")
            return raw, f"recorded_{key}.wav"

    return None, None

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    st.title("Speaker Diarization & Identification")

    # Sidebar inputs
    api_key = st.sidebar.text_input("AssemblyAI API Key", type="password")
    speakers_expected = st.sidebar.slider("Expected speakers", 1, 6, 2)
    threshold = st.sidebar.slider("Similarity threshold", 0.1, 1.0, 0.75, 0.05)

    conv_bytes, conv_name = audio_input("Conversation", "conv")

    st.subheader("Reference Speakers (Optional)")
    ref_files = st.file_uploader("Upload reference WAVs", ["wav"], accept_multiple_files=True)
    references: Dict[str, str | None] = {
        st.text_input(f"Label for {f.name}", key=f.name): None for f in ref_files or []
    }

    if st.button("Analyze"):
        if not conv_bytes:
            st.error("Conversation audio required")
            st.stop()
        if not api_key:
            st.error("AssemblyAI key required")
            st.stop()

        # ── Transcription ────────────────────────────────────────────────
        with st.spinner("Transcribing…"):
            with svc.temp_wav(conv_bytes) as conv_path:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                transcript = loop.run_until_complete(
                    svc.transcribe(conv_path, speakers_expected, api_key)
                )
                segs, rows = svc.segment_by_speaker(conv_path, transcript)
                clip_paths, clip_dir = svc.export_segments(segs)

        # ── Prepare reference temp files ─────────────────────────────────
        ref_dir = pathlib.Path(tempfile.mkdtemp(prefix="refs_"))
        for f in ref_files or []:
            path = ref_dir / f.name
            path.write_bytes(f.getvalue())
            for lbl in references:
                if references[lbl] is None:
                    references[lbl] = str(path)
                    break

        # ── Identification & display ────────────────────────────────────
        model = svc.get_model()
        df = pd.DataFrame(rows, columns=["speaker", "start", "end", "text"])
        df = svc.identify(model, clip_paths, references, df, threshold)

        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "transcript.csv", "text/csv")

        if clip_dir.exists():
            shutil.rmtree(clip_dir, ignore_errors=True)
        if ref_dir.exists():
            shutil.rmtree(ref_dir, ignore_errors=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
