@echo off
echo ============================================
echo  YT-Dubber - Installation
echo ============================================

echo.
echo [1/3] Installing PyTorch with CUDA 12.1 support (RTX 3060)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/3] Installing remaining dependencies...
pip install yt-dlp openai-whisper transformers sentencepiece sacremoses edge-tts pydub google-api-python-client google-auth-oauthlib google-auth-httplib2

echo.
echo [3/3] Installing Whisper (may take a moment)...
pip install git+https://github.com/openai/whisper.git

echo.
echo ============================================
echo  Installation complete!
echo  Run: python dubber.py <youtube_url>
echo ============================================
pause
