# Whisper Clip (Windows + macOS)

Whisper Clip is a tiny, open-source desktop app for Windows and macOS that lets you speak and instantly paste the transcript anywhere. It records your voice, transcribes it locally with OpenAI's open-source Whisper model, and auto-copies the result to your clipboard so you can just use Ctrl+V or Cmd+V in any text box.

If you need a fast, no-fuss way to turn speech into text for notes, prompts, chats, or forms, Whisper Clip keeps it simple and local.

## Features
- One-click record and transcribe
- Auto-copy to clipboard after each recording
- Works offline (audio never leaves your machine)
- Optional transcript panel for reviewing or manual copy
- Runs on Windows 10/11 and macOS 12+

## How it works
Whisper Clip records a short audio clip from your microphone, runs local transcription with Whisper, and copies the text to your clipboard so you can paste it anywhere.

## Requirements
- Windows 10/11 or macOS 12+
- Python 3.10+
- A working microphone

## Setup (Windows)

1) Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

If PyTorch fails to install, use the official PyTorch installer for Windows, then re-run the pip command above.

3) Run the app

```powershell
python app.py
```

## Setup (macOS)

1) Install PortAudio (needed by `sounddevice`)

```bash
brew install portaudio
```

2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

```bash
pip install -r requirements.txt
```

4) Run the app

```bash
python app.py
```

If macOS prompts for microphone access, allow it for your Python interpreter (Terminal / your IDE).

## Behavior
- The transcript box is hidden by default; click the Transcript button to show/hide it.
- Transcriptions are auto-copied to the clipboard after each recording.
- The Copy button re-copies the latest transcript.
- Audio is not saved to disk; it is a one-time in-memory recording only.

## Model selection

Set the `WHISPER_MODEL` environment variable to pick a Whisper model name (for example: `small`, `medium`, `large`). The default is `base`. Larger models are more accurate but slower.

```powershell
$env:WHISPER_MODEL = "small"
python app.py
```

## Notes
- The app loads the model locally and never sends audio to any server.
- Models are stored in the project folder at `model/` once downloaded.
- If your OS prompts for microphone access, allow it for Python.