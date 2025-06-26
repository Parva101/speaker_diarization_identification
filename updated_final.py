import assemblyai as aai
from pydub import AudioSegment
import os
import pandas as pd
from speechbrain.inference import SpeakerRecognition
from dotenv import load_dotenv
import traceback
load_dotenv()
import requests

def validate_audio_file(audio_file):
    """Validate audio file existence and format"""
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if not audio_file.lower().endswith('.wav'):
        raise ValueError("Only WAV files are supported")

def get_transcript(local_audio_path, language):
    try:
        nova2=[]
        sarvam=[]
        deepgram_api= os.getenv('DEEPGRAM_API_KEY')
        token= f"token {deepgram_api}"
        if language=="multilingual":
            url= f"https://api.deepgram.com/v1/listen?smart_format=true&diarize=true&language={language}&model=nova-3&numerals=true&punctuate=true"
        elif language in nova2:
            url= f"https://api.deepgram.com/v1/listen?smart_format=true&diarize=true&model=nova-2&numerals=true&punctuate=true"
        headers = {
    "Authorization": token,
    "Content-Type": "audio/*"
}
        with open(local_audio_path, "rb") as audio_file:
    # Make the HTTP request
           response = requests.post(url, headers=headers, data=audio_file)
        transcript= format_transcript(response)
        return transcript
    except Exception as e:
        raise Exception

def format_transcript(response):
    # 1. Extract all words with speaker attribution
    words = response.json()['results']['channels'][0]['alternatives'][0]['words']

    # 2. Group words by speaker, keeping their order
    speaker_segments = []
    current_speaker = None
    current_segment = []

    for w in words:
        spk = w.get('speaker')
        word = w.get('punctuated_word', w.get('word'))
        # Start a new segment when speaker changes
        if current_speaker is None:
            current_speaker = spk
        if spk != current_speaker:
            # Save current segment
            speaker_segments.append((current_speaker, " ".join(current_segment)))
            # Start new
            current_speaker = spk
            current_segment = [word]
        else:
            current_segment.append(word)

    # Don't forget the last segment!
    if current_segment:
        speaker_segments.append((current_speaker, " ".join(current_segment)))

    transcript_text = ""
    # 3. Write nicely to transcript.txt
    for spk, text in speaker_segments:
        transcript_text += f"Speaker {spk}:\n{text}\n\n"
    
    print("Transcript written to transcript.txt")
    return transcript_text

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

def speaker_identification(speaker_model,output_files,reference_audio,df):
    "This function assigns speaker labels as doctor or patient based on similarity score"
    df_update=df.copy()
    counter=1
    for output_file in output_files:
        speaker_audio = output_file
        score, is_reference = verify_speaker(speaker_model,reference_audio, speaker_audio)
        speaker_label = os.path.splitext(output_file)[0].split("_")[1]
        if is_reference:
            if not doctor_assigned:
                updated_label = ""
                doctor_assigned = True
            # else:
            #     updated_label = f"Doctor {doctor_counter}"
            #     doctor_counter += 1
        else:
            if counter==1:
                updated_label = ""
            else:
                updated_label=f"Patient {counter}"
            counter += 1
        df_update.loc[df_update["Speaker"] == speaker_label, "Speaker"] = updated_label
        print(f"Similarity Score for {output_file}: {score:.2f} - Assigned as {updated_label}")
    return df_update
    
def save_to_csv(df,csv_filename):
    # Save updated transcript
    df.to_csv(csv_filename, index=False)
    print(f"Updated transcript saved to {csv_filename}")

def main():
    """Main function with comprehensive error handling"""
    try:
        audio_file = "Voice 001.wav"
        reference_audio = "Recording.wav"
        csv_filename = "updated_transcript.csv"
        speakers_expected = 3
        for f in [audio_file, reference_audio]:
            validate_audio_file(f)
        transcript = assembly_ai(speakers_expected,audio_file)
        speaker_model=speechbrain_model()
        combined_audio,transcription_data = get_audio_segments(audio_file,transcript)
        if not combined_audio:
            raise ValueError("No audio segments generated")
            
        output_files = export_audio_files(combined_audio)
        
        if not output_files:
            raise ValueError("No audio files exported")
            
        df = pd.DataFrame(transcription_data, 
                         columns=["Speaker", "Start Time", "End Time", "Text"])
        
        df_update = speaker_identification(speaker_model, output_files, reference_audio, df)
        
        try:
            df_update.to_csv(csv_filename, index=False)
            print(f"Transcript saved to {csv_filename}")
        except Exception as e:
            print(f"Failed to save CSV: {str(e)}")

    except Exception as e:
        print(f"Critical Failure: {str(e)}")
        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()