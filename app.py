#!/usr/bin/env python3
"""Local Streamlit app for finding, translating, and rendering short video clips."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from video_clipper import ClipperError, OLLAMA_URL, process_video


ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
OUTPUT_DIR = ROOT / "output_clips"


def _ollama_settings() -> tuple[str, str]:
    try:
        settings = st.secrets["ollama"]
    except (KeyError, StreamlitSecretNotFoundError):
        settings = {}
    if not isinstance(settings, Mapping):
        raise ClipperError("The `[ollama]` Streamlit secrets section must be a table.")
    base_url = str(settings.get("base_url") or os.getenv("OLLAMA_BASE_URL") or OLLAMA_URL).strip()
    api_key = str(settings.get("api_key") or os.getenv("OLLAMA_API_KEY") or "").strip()
    return base_url, api_key


st.set_page_config(
    page_title="Clip Studio",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <!--
    THESIS: Turn a video URL into reviewable vertical clips; avoid opaque creator dashboards.
    OWN-WORLD: Restrained paper-white workspace, ink-dark type, one precise teal action.
    STORY: Paste an authorized source, choose caption language, then preview and download clips.
    FIRST VIEWPORT: A focused title and source form lead; progress and clip previews follow below.
    FORM: Familiar Streamlit workflow, intentionally plain and task-first.
    -->
    <style>
    [data-testid="stAppViewContainer"] { background: #f4f7f8; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: #17252a; letter-spacing: -0.025em; }
    p, label, .stCaption { color: #34464c; }
    [data-testid="stForm"] {
      padding: 1.25rem 1.4rem;
      border: 1px solid #d5e0e2;
      border-radius: 12px;
      background: #ffffff;
    }
    [data-testid="stVideo"] { border-radius: 8px; overflow: hidden; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
      min-height: 2.8rem;
      border-radius: 8px;
      font-weight: 650;
    }
    div[data-testid="stDownloadButton"] > button {
      min-height: 2.5rem;
      border-radius: 7px;
    }
    :focus-visible { outline: 3px solid #087e6a; outline-offset: 2px; }
    @media (max-width: 700px) {
      .block-container { padding: 1.25rem 1rem 3rem; }
      [data-testid="stForm"] { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Clip Studio")
st.markdown(
    "Find the moments worth sharing. **Download, transcribe, translate, and render locally.**"
)
st.caption(
    "The video is downloaded from its source URL. Transcription and rendering run on this "
    "app's host; Ollama must be reachable from that host."
)

with st.form("clip_generation"):
    video_url = st.text_input(
        "Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a public video URL supported by yt-dlp.",
    )
    left, right = st.columns([1, 1])
    with left:
        subtitle_language = st.selectbox(
            "Subtitle language",
            ("English", "Japanese", "Chinese (Simplified)"),
            help=(
                "English captions use the transcript directly. Japanese and Chinese captions "
                "are translated locally by Ollama from the English transcript."
            ),
        )
    with right:
        st.markdown(
            '<div style="height: 1.8rem"></div>'
            '<p style="font-size: 0.9rem; margin: 0.35rem 0 0">'
            "Moment analysis uses Ollama llama3.2. Cloud deployments need a hosted endpoint "
            "in Streamlit Secrets.</p>",
            unsafe_allow_html=True,
        )
    has_permission = st.checkbox(
        "I have permission to download and repurpose this video.",
        help="Only use sources you own or are authorized to use.",
    )
    submitted = st.form_submit_button(
        "Generate Unlimited Clips",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not video_url.strip():
        st.error("Paste a video URL to get started.")
    elif not has_permission:
        st.error("Confirm that you have permission to download and repurpose this video.")
    else:
        progress_bar = st.progress(0, text="Preparing services…")
        status = st.status("Preparing services…", expanded=True)

        def update_progress(label: str, completed: int, total: int) -> None:
            fraction = completed / total if total else 0
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=label)
            status.update(label=label, state="running", expanded=True)

        try:
            ollama_base_url, ollama_api_key = _ollama_settings()
            with status:
                result = process_video(
                    video_url,
                    DOWNLOADS_DIR,
                    OUTPUT_DIR,
                    progress=update_progress,
                    subtitle_language=subtitle_language,
                    ollama_base_url=ollama_base_url,
                    ollama_api_key=ollama_api_key,
                )
            st.session_state["clip_results"] = [
                {
                    "video_path": str(clip.video_path),
                    "subtitle_path": str(clip.subtitle_path),
                    "start": clip.start,
                    "end": clip.end,
                    "reason": clip.reason,
                }
                for clip in result.clips
            ]
            st.session_state["clip_summary"] = {
                "word_count": result.word_count,
                "transcript_path": str(result.transcript_path),
                "rejected_moments": result.rejected_moments,
                "subtitle_language": result.subtitle_language,
            }
            if result.clips:
                status.update(
                    label=f"Ready: {len(result.clips)} clip(s)",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label="No qualifying moments found",
                    state="complete",
                    expanded=False,
                )
            st.rerun()
        except (ClipperError, ValueError, OSError) as error:
            status.update(label="Processing stopped", state="error", expanded=True)
            st.error(str(error))


def _format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds_part = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds_part:02}" if hours else f"{minutes}:{seconds_part:02}"


clips = st.session_state.get("clip_results", [])
summary = st.session_state.get("clip_summary")

st.divider()
if clips and summary:
    st.subheader(f"{len(clips)} clip(s) ready")
    st.caption(
        f"{summary['word_count']:,} timestamped words · "
        f"{summary['subtitle_language']} subtitles · saved in output_clips/"
    )
    if summary["rejected_moments"]:
        st.warning(
            f"Ollama returned {summary['rejected_moments']} invalid or out-of-range "
            "moment suggestion(s); they were skipped."
        )
    st.download_button(
        "Download timestamped transcript (JSON)",
        data=Path(summary["transcript_path"]).read_bytes(),
        file_name=Path(summary["transcript_path"]).name,
        mime="application/json",
    )
    for offset in range(0, len(clips), 2):
        columns = st.columns(2)
        for column, clip in zip(columns, clips[offset : offset + 2]):
            video_path = Path(clip["video_path"])
            subtitle_path = Path(clip["subtitle_path"])
            with column:
                with st.container(border=True):
                    st.markdown(
                        f"**{_format_time(clip['start'])}–{_format_time(clip['end'])}**"
                    )
                    st.text(clip["reason"] or "Selected as a self-contained moment.")
                    st.video(str(video_path))
                    st.download_button(
                        "Download clip",
                        data=video_path.read_bytes(),
                        file_name=video_path.name,
                        mime="video/mp4",
                        key=f"video-{video_path.name}",
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download subtitles",
                        data=subtitle_path.read_bytes(),
                        file_name=subtitle_path.name,
                        mime="application/x-subrip",
                        key=f"subtitle-{subtitle_path.name}",
                        use_container_width=True,
                    )
elif summary:
    st.info(
        "The video was transcribed, but no 30–60 second moments met the local model’s "
        "selection criteria. Try another source with more spoken content."
    )
else:
    st.info(
        "Paste a video URL above. Finished clips will appear here with previews, "
        "download buttons, and subtitle files."
    )
