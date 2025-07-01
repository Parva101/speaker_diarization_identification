import assemblyai as aai
from pydub import AudioSegment
import os
import pandas as pd
from speechbrain.inference import SpeakerRecognition
from dotenv import load_dotenv
import traceback
load_dotenv()

def validate_audio_file(audio_file):
    """Validate audio file existence and format"""
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if not audio_file.lower().endswith('.wav'):
        raise ValueError("Only WAV files are supported")
    
def assembly_ai(speakers_expected: int, audio_file: str, api_key: str):
    """
    Transcribe audio using AssemblyAI with speaker labels.
    """
    try:
        aai.settings.api_key = api_key
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=speakers_expected
        )
        transcript = aai.Transcriber().transcribe(audio_file, config)
        return transcript
    except Exception as e:
        print(f"AssemblyAI Error: {str(e)}")
        raise

def speechbrain_model():
    # Initialize SpeakerRecognition model
    try:
        speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp/embedding_model"
        )
        return speaker_model
    except Exception as e:
        print(f"Model Loading Error: {str(e)}")
        raise

def get_audio_segments(audio_file: str, transcript):
    """
    Function to get audio segments for each speaker
    Returns:
        combined_audio: dict[str, AudioSegment]
        transcription_data: list
    """
    try:
        # Load audio file
        audio = AudioSegment.from_wav(audio_file)

        # Combine audio for each speaker
        combined_audio = {}
        transcription_data = []
        for utterance in transcript.utterances:
            try:
                start_time = int(utterance.start * 1000)  # in ms
                end_time = int(utterance.end * 1000)  # in ms
                speaker = str(utterance.speaker)
                segment = audio[start_time:end_time]

                if speaker not in combined_audio:
                    combined_audio[speaker] = segment
                else:
                    combined_audio[speaker] += segment

                transcription_data.append([
                    speaker,
                    start_time,
                    end_time,
                    utterance.text
                ])
            except Exception as e:
                print(f"Error processing utterance: {str(e)}")
                continue

        return combined_audio, transcription_data
    except Exception as e:
        print(f"Audio Processing Error: {str(e)}")
        raise

def export_audio_files(combined_audio: dict[str, AudioSegment]) -> dict[str, str]:
    """Export combined audio for each speaker"""
    output_files = {}
    for speaker, audio_segment in combined_audio.items():
        try:
            output_file = f"speaker_{speaker}.wav"
            audio_segment.export(output_file, format="wav")
            output_files[speaker] = output_file
            print(f"Saved audio for Speaker {speaker} to {output_file}")
        except Exception as e:
            print(f"Failed to export {output_file}: {str(e)}")
            continue
    return output_files

def verify_speaker(speaker_model, audio1, audio2, threshold=0.5):
    """Speaker verification with validation using encode_file for single files"""
    try:
        for f in [audio1, audio2]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Audio file missing: {f}")
        
        emb1 = speaker_model.encode_file(audio1)
        emb2 = speaker_model.encode_file(audio2)
        score = speaker_model.similarity(emb1, emb2)
        return score.item()
    except Exception as e:
        print(f"Verification Error: {str(e)}")
        return 0.0

def speaker_identification(speaker_model, output_files: dict[str, str], reference_paths: dict[str, str], df, threshold=0.5):
    """
    Labels each speaker clip based on the best-matching reference audio.
    If no reference paths are provided, returns df with generic speaker labels.

    Args:
        speaker_model: SpeechBrain ECAPA model
        output_files: Dict mapping speaker IDs (str) to file paths
        reference_paths: Dict mapping names to file paths for known speakers
        df: DataFrame with a 'speaker' column
        threshold: Similarity score threshold

    Returns:
        DataFrame with 'speaker' column labeled
    """
    if not reference_paths or not any(reference_paths.values()):
        df['speaker'] = df['speaker'].apply(lambda x: f"Speaker {x}")
        return df

    label_map = {}
    for speaker_id, spk_file in output_files.items():
        best_score = 0.0
        best_label = "Unknown"
        for ref_name, ref_file in reference_paths.items():
            if not ref_file:  # Skip if no path
                continue
            try:
                score = verify_speaker(speaker_model, spk_file, ref_file)
                if score > best_score:
                    best_score = score
                    best_label = ref_name
            except Exception:
                continue
        label_map[str(speaker_id)] = best_label if best_score >= threshold else "Unknown"

    df['speaker'] = df['speaker'].map(lambda x: label_map.get(str(x), "Unknown"))
    return df
    
def save_to_csv(df, csv_filename):
    # Save updated transcript
    df.to_csv(csv_filename, index=False)
    print(f"Updated transcript saved to {csv_filename}")

# def main():
#     """Main function with comprehensive error handling"""
#     try:
#         audio_file = "Voice 001.wav"
#         reference_audio = "Recording.wav"
#         csv_filename = "updated_transcript.csv"
#         speakers_expected = 3
#         for f in [audio_file, reference_audio]:
#             validate_audio_file(f)
#         transcript = (speakers_expected,audio_file)
#         speaker_model=speechbrain_model()
#         combined_audio,transcription_data = get_audio_segments(audio_file,transcript)
#         if not combined_audio:
#             raise ValueError("No audio segments generated")
            
#         output_files = export_audio_files(combined_audio)
        
#         if not output_files:
#             raise ValueError("No audio files exported")
            
#         df = pd.DataFrame(transcription_data, 
#                          columns=["Speaker", "Start Time", "End Time", "Text"])
        
#         df_update = speaker_identification(speaker_model, output_files, reference_audio, df)
        
#         try:
#             df_update.to_csv(csv_filename, index=False)
#             print(f"Transcript saved to {csv_filename}")
#         except Exception as e:
#             print(f"Failed to save CSV: {str(e)}")

#     except Exception as e:
#         print(f"Critical Failure: {str(e)}")
#         print(traceback.format_exc())
#         exit(1)


# if __name__ == "__main__":
#     main()