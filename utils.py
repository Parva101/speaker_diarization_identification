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
    
def assembly_ai(speakers_expected,audio_file,api_key):
    """
    function to set Assembly AI parameters
    """
    try:
        aai.settings.api_key = api_key
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=speakers_expected
        )
        transcript = aai.Transcriber().transcribe(audio_file, config) #only transccript without speaker labels
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

def format_time(milliseconds):
    """
    Function to format time in mm:ss for the transcript since, Assembly ai gives timestamp in milliseconds
    """
    seconds = milliseconds // 1000
    minutes = seconds // 60
    return f"{minutes}:{seconds % 60:02}"

def get_audio_segments(audio_file,transcript):
    """
    Function to get audio segments for each speaker
    """
    try:
    # Load audio file
        audio = AudioSegment.from_wav(audio_file)

    # Combine audio for each speaker
        combined_audio = {}
        transcription_data= []
        for utterance in transcript.utterances:
            try:
                start_time = utterance.start
                end_time = utterance.end
                speaker = utterance.speaker
                segment = audio[start_time:end_time]
                
                if speaker not in combined_audio:
                    combined_audio[speaker] = segment
                else:
                    combined_audio[speaker] += segment
                    
                transcription_data.append([
                    speaker,
                    format_time(start_time),
                    format_time(end_time),
                    utterance.text
                ])
            except Exception as e:
                print(f"Error processing utterance: {str(e)}")
                continue
                
        return combined_audio, transcription_data
    except Exception as e:
        print(f"Audio Processing Error: {str(e)}")
        raise

def export_audio_files(combined_audio):
    """Export combined audio for each speaker"""
    output_files = []
    for speaker, audio_segment in combined_audio.items():
        try:
            output_file = f"speaker_{speaker}.wav"
            audio_segment.export(output_file, format="wav")
            output_files.append(output_file)
            print(f"Saved audio for Speaker {speaker} to {output_file}")
        except Exception as e:
            print(f"Failed to export {output_file}: {str(e)}")
            continue
    return output_files

def verify_speaker(speaker_model,audio1, audio2, threshold=0.5):
    """Speaker verification with validation"""
    try:
        for f in [audio1, audio2]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Audio file missing: {f}")
                
        emb1 = speaker_model.encode_batch(speaker_model.load_audio(audio1))
        emb2 = speaker_model.encode_batch(speaker_model.load_audio(audio2))
        score = speaker_model.similarity(emb1, emb2)
        return score.item(), score >= threshold
    except Exception as e:
        print(f"Verification Error: {str(e)}")
        return 0.0, False

def speaker_identification(speaker_model,output_files,reference_paths,df, threshold=0.5):
    """
    Labels each speaker clip based on the best-matching reference audio.
    If no reference paths are provided, returns df with generic speaker labels.

    Args:
        model: SpeechBrain ECAPA model
        output_files: Dict mapping speaker IDs to file paths
        reference_paths: Dict mapping names to file paths for known speakers
        df: DataFrame with a 'speaker' column
        threshold: Similarity score threshold

    Returns:
        DataFrame with 'speaker' column labeled
    """
    if not reference_paths:
        # No references provided: generic labels
        df_copy = df.copy()
        df_copy['speaker'] = df_copy['speaker'].apply(lambda s: f"Speaker {s}")
        return df_copy

    label_map = {}
    for speaker_id, spk_file in output_files.items():
        best_score = 0.0
        best_label = "Unknown"
        for ref_name, ref_file in reference_paths.items():
            try:
                score = verify_speaker(speaker_model, spk_file, ref_file)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_label = ref_name
        label_map[speaker_id] = best_label if best_score >= threshold else "Unknown"

    df['speaker'] = df['speaker'].map(label_map).fillna("Unknown")
    return df
    
def save_to_csv(df,csv_filename):
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