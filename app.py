import streamlit as st
import tempfile
import os
import time
import pandas as pd
import io
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder
from utils import (
    validate_audio_file,
    speechbrain_model,
    get_audio_segments,
    export_audio_files,
    speaker_identification,
    save_to_csv,
    assembly_ai
)

# Streamlit app configuration
st.set_page_config(page_title="Speaker Diarization & Identification")

def save_audio_file(audio_bytes, suffix=".wav"):
    """Save raw audio bytes to a temporary WAV file and return its path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return None

def audio_input(label: str, key: str) -> tuple[bytes, str]:
    """Presents upload and recording tabs. Returns (audio_bytes, filename)."""
    st.subheader(label)
    tab_up, tab_rec = st.tabs(["Upload", "Record"])
        
    with tab_up:
        uploader = st.file_uploader(
            f"Upload {label} (WAV)",
            type=["wav"],
            key=f"upload_{key}"
        )
        if uploader:
            return uploader.getvalue(), uploader.name
    
    with tab_rec:
        recorded = audio_recorder(
            text="Click to start/stop recording",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=300,  # 5 minutes threshold
            energy_threshold=(-1.0, 1.0),
            key=f"manual_recorder_{key}"
        )
        if recorded:
            st.audio(recorded, format="audio/wav")
            # Load into pydub and read length in ms
            audio_seg = AudioSegment.from_file(io.BytesIO(recorded), format="wav")
            duration_seconds = len(audio_seg) / 1000.0
            st.caption(f"Recorded duration: {duration_seconds:.2f} s")  
            return recorded, f"recorded_{key}.wav"  
    return None,None

def main():
    st.title("Speaker Diarization & Identification")
    st.markdown("""
    **Speaker Diarization & Identification** is a Streamlit app that processes
    audio recordings to:

    1. **Load or Record WAV Audio** directly in the browser  
    2. **Transcribe Speech**  
    3. **Diarize Speakers**—split the conversation into speaker turns  
    4. **Identify Speakers** by clustering voice prints or reference samples  
    5. **Export Clips & Transcripts** per speaker for downstream analysis  """)
    st.sidebar.header("Settings")
    speakers_expected = st.sidebar.slider("Expected Number of Speakers", 1, 6, 2)

    #get user's deepgram api key
    api_key=st.sidebar.text_input("AssemblyAI API Key",type="password" )

    # Audio input sections
    conv_bytes, conv_name = audio_input("Conversation Audio", "conv")
    st.subheader("Reference Audios & Names")

    ref_files = st.file_uploader("Upload reference audio files (WAV) (Optional)" \
    "**You can upload multiple reference files as well", type=["wav"], accept_multiple_files=True)
    references = {}
    for file in ref_files:
        name = st.text_input(f"Name for {file.name}", key=file.name)
        if name:
            references[name] = file

    if st.button("Analyze Conversation", type="primary"):
        # Process audio inputs
        try:
            # Handle conversation audio
            if not conv_bytes or not api_key:
                st.error("Provide conversation audio")
                return
            if not api_key:
                st.error("Provide your AssemblyAI API key.")
                return
            conv_path = save_audio_file(conv_bytes, suffix=os.path.splitext(conv_name)[1])
            if not conv_path:
                return

            try:
                validate_audio_file(conv_path)
            except Exception as e:
                st.error(str(e))
                return
            
            # Handle reference audio
            for file in ref_files:
                # get value and filename
                val = file.getvalue()
                path = save_audio_file(val, suffix=os.path.splitext(file.name)[1])
                try:
                    validate_audio_file(path)
                    # find matching key by filename
                    for label in list(references):
                        if file.name in label or references[label] is None:
                            references[label] = path
                            break
                except Exception:
                    if path and os.path.exists(path):
                        os.remove(path)

            with st.spinner("Analyzing audio..."):
                # Transcription pipeline
                start_time = time.time()
                transcript = assembly_ai(speakers_expected,conv_path,api_key)
                st.success(f"Transcription completed in {time.time()-start_time:.1f}s")

                # Audio processing
            combined_audio, transcript_data = get_audio_segments(conv_path, transcript)
            os.remove(conv_path)
            output_files = export_audio_files(combined_audio)

            # Create transcript dataframe
            df = pd.DataFrame(transcript_data,
                            columns=["speaker","start","end","text"])

            # Speaker identification
            model = speechbrain_model()
            df_identified = speaker_identification(
                model, output_files, references, df
            )

            # Display results
            st.dataframe(df_identified)

            # Download options
            csv = df_identified.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Transcript",
                data=csv,
                file_name="transcript.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            st.stop()
        
        finally:
            # Cleanup temporary files
            cleanup_paths = list(references.values())
            if output_files is not None:
                if isinstance(output_files, dict):
                    cleanup_paths += list(output_files.values())
                else:
                    cleanup_paths += output_files
            for path in cleanup_paths:
                if isinstance(path, str) and os.path.exists(path):
                    os.remove(path)

if __name__ == "__main__":
    main()