# Video Clip Studio

This Streamlit app downloads a single video URL, transcribes English speech
with faster-whisper `base.en`, asks an Ollama `llama3.2` model to identify every
qualifying 30–60 second moment, and renders 1080x1920 clips with bold, outlined
captions. Japanese and Simplified Chinese subtitles are translated by Ollama;
English subtitles use the original transcript.

Use the downloader only for videos you own or have permission to download and
repurpose. Downloads are stored in `downloads/`, and generated MP4/SRT files
are stored in `output_clips/`. When hosted, both directories are on the server
and are ephemeral; download results from the browser before the app restarts.

## Run locally on macOS

Install the system tools and start the local model:

```bash
brew install ffmpeg ollama
ollama pull llama3.2
```

In this folder, create a Python environment and install the app dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-video-clipper.txt
```

Start Ollama if it is not already running, then launch the app:

```bash
ollama serve
streamlit run app.py --server.address 127.0.0.1
```

Open the localhost URL printed by Streamlit. Paste an authorized video URL,
choose English, Japanese, or Simplified Chinese subtitles, confirm your rights
to use the video, and select **Generate Unlimited Clips**. The first run may
take longer while faster-whisper downloads its `base.en` model.

`base.en` is intended for English speech. Japanese and Chinese are subtitle
translation targets; source audio is still transcribed in English before the
local Ollama translation step.

## Deploy to Streamlit Community Cloud

The repository includes the root `requirements.txt` and `packages.txt` files
Community Cloud uses to install Python packages and ffmpeg. In the Streamlit
Cloud workspace, create an app from this repository and branch, set `app.py` as
the entrypoint, and use Python 3.11 to match the local environment.

Community Cloud cannot reach Ollama running on your Mac. Provide an Ollama API
endpoint reachable from the deployed app in the app's **Secrets** settings:

```toml
[ollama]
base_url = "https://your-ollama-host.example"
api_key = "your-api-key"
```

The API key is optional for an endpoint that does not require authentication.
Do not commit real credentials; `.streamlit/secrets.toml` is ignored by Git.
Use a secured HTTPS endpoint and restrict its access to the deployment.

Community Cloud hosts apps in the United States and runs this app's downloads,
speech recognition, and video rendering on its own compute resources. Do not
send private videos there, and expect long videos or many output clips to need
more resources than a local Mac. The deployed filesystem is temporary, so
download finished clips before the app restarts.

Review the app's visibility and access settings before deployment. Anyone who
can access a public app can submit URLs and consume its compute resources.
