import streamlit as st
import os
import shutil
import librosa
import soundfile as sf
import yt_dlp
from yt_dlp.utils import DownloadError
from dotenv import load_dotenv
import whisper
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize Claude API
if not anthropic_api_key:
    raise ValueError("Anthropic API key is not set in the environment variables.")
anthropic = Anthropic(api_key=anthropic_api_key)

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file_path):
        audio = whisper.load_audio(audio_file_path)
        result = self.model.transcribe(audio)
        return result['text']

def summarize_claude(chunks: list[str], system_prompt: str, model="claude-2", output_file=None):
    summaries = []
    for chunk in chunks:
        prompt = f"{system_prompt}\n\n{chunk}"
        completion = anthropic.completions.create(
            model=model,
            max_tokens_to_sample=300,  # Ensure this parameter is present
            prompt=prompt
        )
        summary = completion['completion']  # Use the correct key for accessing the text
        summaries.append(summary)
        
    if output_file is not None:
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")
    
    return summaries


def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with yt_dlp.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        pass  # Handle error as needed

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename

def find_audio_files(path, extension=".mp3"):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))
    return audio_files

def chunk_audio(filename, segment_length: int, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    audio, sr = librosa.load(filename, sr=44100)
    duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(duration / segment_length) + 1
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)
    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)

def summarize_youtube_video(youtube_url, outputs_dir, progress_bar, progress_text, summarization_function):
    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks/"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    segment_length = 10 * 60  # 10 minutes

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
        os.mkdir(outputs_dir)

    progress_text.text("Downloading video...")
    audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)
    progress_bar.progress(0.25)

    progress_text.text("Chunking audio...")
    chunked_audio_files = chunk_audio(audio_filename, segment_length=segment_length, output_dir=chunks_dir)
    progress_bar.progress(0.5)

    progress_text.text("Transcribing audio...")
    transcriber = WhisperTranscriber()
    transcriptions = [transcriber.transcribe(file) for file in chunked_audio_files]
    with open(transcripts_file, "w") as file:
        for transcript in transcriptions:
            file.write(transcript + "\n")
    progress_bar.progress(0.75)

    progress_text.text("Generating summary...")
    system_prompt = "You are a helpful assistant that summarizes and distills YouTube videos. You are provided chunks of raw audio that were transcribed from the video's audio. Summarize and distill the current chunk to succinct and clear bullet points of its contents."
    summaries = summarization_function(transcriptions, system_prompt=system_prompt, output_file=summary_file)

    system_prompt_tldr = "You are a helpful assistant that summarizes YouTube videos. Someone has already summarized the video to key points. Summarize the key points to one or two sentences that capture the essence of the video."
    long_summary = "\n".join(summaries)
    short_summary = summarization_function([long_summary], system_prompt=system_prompt_tldr, output_file=summary_file)[0]

    progress_bar.progress(1.0)
    progress_text.text("Summary complete.")

    return long_summary, short_summary

def main():
    st.title("AI BOT")

    summarization_choice = st.sidebar.selectbox("Choose summarization method:", ["Claude"])
    summarization_function = summarize_claude

    youtube_url = st.text_input("Enter meeting URL:", "")

    if st.button("Summarize Video"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        with st.spinner("Summarizing... This might take a while."):
            outputs_dir = "outputs/"
            long_summary, short_summary = summarize_youtube_video(youtube_url, outputs_dir, progress_bar, progress_text, summarization_function)

        st.subheader("Long Summary:")
        st.write(long_summary)

        st.subheader("Video - TL;DR")
        st.write(short_summary)

if __name__ == "__main__":
    main()
