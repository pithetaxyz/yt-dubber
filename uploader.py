"""
YouTube uploader using OAuth 2.0.
First run: opens browser to authenticate with your Google account.
Credentials are cached in token.pickle for subsequent runs.
"""

import os
import sys
import json
import pickle
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS = Path(__file__).parent / "client_secrets.json"
TOKEN_CACHE = Path(__file__).parent / "token.pickle"


def get_youtube_client():
    creds = None

    if TOKEN_CACHE.exists():
        with open(TOKEN_CACHE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRETS.exists():
                raise FileNotFoundError(
                    "\n[!] client_secrets.json not found.\n"
                    "    Follow these steps:\n"
                    "    1. Go to https://console.cloud.google.com\n"
                    "    2. Create a project → Enable 'YouTube Data API v3'\n"
                    "    3. Credentials → Create OAuth 2.0 Client ID (Desktop app)\n"
                    "    4. Download JSON → save as client_secrets.json in this folder\n"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_CACHE, "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


YOUTUBE_DESCRIPTION_LIMIT = 5000


def build_description(source_title: str, source_url: str, transcript_path: str) -> str:
    desc = (
        "This video has been AI-dubbed from Chinese to English for educational purposes, "
        "to help spread knowledge and make this content accessible to a wider audience. "
        "All credit goes to the original creator.\n"
    )
    if source_title:
        desc += f"\nOriginal title: {source_title}"
    if source_url:
        desc += f"\nOriginal video: {source_url}"

    if transcript_path and Path(transcript_path).exists():
        try:
            saved = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
            segments = saved["segments"] if isinstance(saved, dict) else saved
            transcript_lines = [
                f"[{int(s['start']//60):02d}:{int(s['start']%60):02d}] {s.get('text', '')}"
                for s in segments
            ]
            transcript_text = "\n\nTranscript:\n" + "\n".join(transcript_lines)
            if len(desc) + len(transcript_text) <= YOUTUBE_DESCRIPTION_LIMIT:
                desc += transcript_text
            else:
                # Fit as many lines as possible
                available = YOUTUBE_DESCRIPTION_LIMIT - len(desc) - len("\n\nTranscript:\n") - len("\n[truncated]")
                fitted = []
                used = 0
                for line in transcript_lines:
                    if used + len(line) + 1 > available:
                        break
                    fitted.append(line)
                    used += len(line) + 1
                if fitted:
                    desc += "\n\nTranscript:\n" + "\n".join(fitted) + "\n[truncated]"
        except Exception:
            pass

    return desc


def upload_to_youtube(
    video_path: str,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    privacy: str = "private",
    source_url: str = "",
    source_title: str = "",
    transcript_path: str = "",
) -> str:
    print("\n--- Uploading to YouTube ---")
    youtube = get_youtube_client()

    if not description:
        description = build_description(source_title, source_url, transcript_path)

    body = {
        "snippet": {
            "title": title[:100],  # YouTube title limit
            "description": description,
            "tags": tags or ["dubbed", "chinese", "english", "AI dub"],
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, mimetype="video/mp4", chunksize=10 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  Uploading... {pct}%", end="\r")

    video_id = response["id"]
    print(f"\n  Upload complete!")
    print(f"  Video ID : {video_id}")
    print(f"  URL      : https://www.youtube.com/watch?v={video_id}")
    print(f"  Status   : {privacy} - change visibility in YouTube Studio if needed")
    return video_id


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python uploader.py <video.mp4> <title>")
        sys.exit(1)
    upload_to_youtube(sys.argv[1], sys.argv[2])
