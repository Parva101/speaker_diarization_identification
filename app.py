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
    """Save audio bytes to a temporary file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return None

def audio_input(label, key):
    """Audio input component with manual stop control"""
    st.subheader(label)
    tab1, tab2 = st.tabs(["Upload", "Record"])
    
    audio_file = None
    audio_bytes = None
    
    with tab1:
        audio_file = st.file_uploader(
            f"Upload {label} (WAV)",
            type=["wav"],
            key=f"upload_{key}"
        )
        if audio_file:
            return audio_file.getvalue(), audio_file.name
    
    with tab2:
        audio_bytes = audio_recorder(
            text="Click to start/stop recording",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=300,  # 5 minutes threshold
            energy_threshold=(-1.0, 1.0),
            key=f"manual_recorder_{key}"
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            # Load into pydub and read length in ms
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            duration_seconds = len(audio_seg) / 1000.0
            st.caption(f"Recorded duration: {duration_seconds:.2f} s")    
    return audio_file, audio_bytes

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
    conv_file, conv_recording = audio_input("Conversation Audio", "conv")
    st.subheader("Reference Audios & Names")
    ref_files = st.file_uploader("Upload reference audio files (WAV)" \
    "**You can upload multiple reference files as well", type=["wav"], accept_multiple_files=True)
    reference_info = []
    for ref in ref_files:
        ref_name = st.text_input(f"Name for {ref.name}", key=ref.name)
        if ref_name:
            reference_info.append((ref_name, ref.getvalue(), ref.name))

    if st.button("Analyze Conversation", type="primary"):
        # Process audio inputs
        try:
            # Handle conversation audio
            if conv_file:
                conv_path = save_audio_file(conv_file.getvalue())
            elif conv_recording:
                conv_path = save_audio_file(conv_recording)
            else:
                st.error("Missing conversation audio input")
                return

            # Handle reference audio
            reference_paths = {}
            for name, data, fname in reference_info:
                path = save_audio_file(data, suffix=os.path.splitext(fname)[1])
                if path and validate_audio_file(path):
                    reference_paths[name] = path

            with st.spinner("Analyzing audio..."):
                # Validate files
                validate_audio_file(conv_path)

                # Transcription pipeline
                start_time = time.time()
                transcript = assembly_ai(speakers_expected,conv_path,api_key)
                st.info(f"Transcription completed in {time.time()-start_time:.1f}s")

                # Audio processing
                combined_audio, transcript_data = get_audio_segments(conv_path, transcript)
                output_files = export_audio_files(combined_audio)

                # Create transcript dataframe
                df = pd.DataFrame(transcript_data,
                                columns=["Speaker", "Start Time", "End Time", "Text"])

                # Speaker identification
                model = speechbrain_model()
                df_identified = speaker_identification(
                    model, output_files, reference_paths, df
                )

                # Display results
                st.success("Analysis Complete!")
                st.dataframe(df_identified.style.set_properties(**{
                    'text-align': 'left',
                    'white-space': 'pre-wrap'
                }))

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
            for f in [conv_path] + list(reference_paths.values()) + list(output_files.values()):
                if f and os.path.exists(f):
                    os.remove(f)

if __name__ == "__main__":
    main()