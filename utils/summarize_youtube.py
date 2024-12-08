# utils/summarize_youtube.py

from utils.text_generation import summarize_text
import youtube_dl  # Ensure youtube_dl is installed: pip install youtube_dl
import logging

def summarize_youtube_video(youtube_url):
    """
    Downloads the transcript of a YouTube video and summarizes it.
    """
    try:
        # Extract video information
        with youtube_dl.YoutubeDL({'skip_download': True}) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_title = info_dict.get('title', None)

        # Attempt to get the transcript using youtube_transcript_api
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

        video_id = info_dict.get('id', None)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Choose the English transcript
        transcript = transcript_list.find_transcript(['en'])
        transcript_data = transcript.fetch()

        # Combine the transcript text
        full_text = " ".join([entry['text'] for entry in transcript_data])

        # Summarize the transcript
        summary = summarize_text(full_text, max_length=300)
        return summary

    except TranscriptsDisabled:
        logging.error(f"Transcripts are disabled for video: {youtube_url}")
        return "Transcripts are disabled for this YouTube video."
    except NoTranscriptFound:
        logging.error(f"No transcript found for video: {youtube_url}")
        return "No transcript found for this YouTube video."
    except Exception as e:
        logging.error(f"Error summarizing YouTube video {youtube_url}: {e}")
        return f"An error occurred while summarizing the video: {e}"
