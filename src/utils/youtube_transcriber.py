"""
Utils function to get youtube transcripts (Internal Tool for now)
This API kinda suck!
Need to find a better one later on to process data for the RAG.
"""

import re
import os
import logging
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    pattern = re.compile(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*")
    match = pattern.search(url)
    if match:
        return match.group(1)
    return None


def get_transcript_and_save(video_url, file_path):
    """
    Extract the transcript from a YouTube video and save it to a text file.

    Args:
        video_url (str): The URL of the YouTube video.
        file_path (str): The path to the file where the transcript will be saved.
    """
    # Ensure the directory for the file path exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directory '{directory}' created.")

    video_id = extract_video_id(video_url)
    if video_id:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        with open(file_path, "w") as f:
            for entry in transcript:
                f.write(entry["text"] + "\n")
    else:
        logging.info("Failed to extract video ID from the URL.")


def main():
    """Run main for testing"""
    # Example usage
    # url = "https://www.youtube.com/watch?v=1qw5ITr3k9E&ab_channel=freeCodeCamp.org"
    # f_path = r"Documents/mock interview.txt"

    url = "https://www.youtube.com/watch?v=p5O-_AiKD_Q"
    f_path = r"Documentation/Llama 3 8B local agents tutorial.txt"

    get_transcript_and_save(url, f_path)


if __name__ == "__main__":
    main()
