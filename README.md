
# Speaker Diarization \& Identification App

A **Streamlit web app** for speaker diarization and identification in audio files. Upload or record audio, transcribe conversations, and identify speakers using reference samples. Powered by AssemblyAI and SpeechBrain.

## Features

- **Speaker Diarization:** Automatically segments audio by speaker.
- **Speaker Identification:** Match speakers to reference samples using embeddings.
- **Audio Upload \& Recording:** Upload WAV files or record directly in the browser.
- **Interactive UI:** Built with Streamlit for easy use.
- **Downloadable Results:** Export diarized and identified transcripts as CSV.


## Demo
https://speaker-diarization-identification.streamlit.app/
1. Upload or record a conversation audio (WAV format).
2. (Optional) Upload reference audio samples for known speakers.
3. Set the expected number of speakers and similarity threshold.
4. Enter your AssemblyAI API key.
5. Click **Analyze** to process the audio.
6. View and download the diarized transcript.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Parva101/speaker_diarization_identification.git
cd speaker_diarization_identification
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the provided local URL in your browser.

## Configuration

- **AssemblyAI API Key:** Required for transcription. Get your key from [AssemblyAI](https://www.assemblyai.com/).
- **Expected Speakers:** Set the number of speakers in the sidebar.
- **Similarity Threshold:** Adjust to control strictness of speaker matching.



## Technologies Used

- **Streamlit:** Web app framework
- **AssemblyAI:** Speech-to-text API
- **SpeechBrain:** Speaker embedding and recognition
- **PyDub:** Audio processing
- **Pandas:** Data handling


## Example

| Speaker | Start (s) | End (s) | Text |
| :-- | :-- | :-- | :-- |
| John | 0.0 | 5.2 | Hello, how are you? |
| Jane | 5.2 | 8.7 | I'm good, thank you! |

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [AssemblyAI](https://www.assemblyai.com/)
- [SpeechBrain](https://speechbrain.github.io/)
- [Streamlit](https://streamlit.io/)


<div style="text-align: center">⁂</div>

[^1]: app.py

[^2]: config.py

[^3]: requirements.txt

[^4]: utils.py

