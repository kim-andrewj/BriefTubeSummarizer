!pip install streamlit -q

!pip install deepface

!pip install yt-dlp openai-whisper -q

!pip install gdown

#!gdown --id 13CgQPPgMqyKKZ82RDe3Oloq5zCMHURsx --output BriefTube.png
#
# !gdown --id 13YZm82frO4-XCwmetJjuyCR0KLpw3ziv --output BriefTube_streamlit.png

! wget -q -O - ipv4.icanhazip.com


# most updated version

%%writefile app.py
import streamlit as st
import subprocess
import os
import whisper
import uuid
import nltk
import cv2
import math
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI

# Download necessary resources
nltk.download("punkt_tab")
client = OpenAI(api_key=INSERTAPIKEYHERE)

# Try to import DeepFace for emotion detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    st.warning("⚠️ DeepFace not installed. Facial emotion detection will be skipped.")

# Load BLIP model once and cache it
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Function to send prompts to OpenAI
def get_completion(prompt, model="gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# Ask GPT to segment transcript into sections with time ranges
def gpt_segment_transcript(transcript, video_duration_sec=600):
    prompt = f"""
Break this YouTube video transcript into ~5 segments. For each, include:
- a short title
- a one-sentence description
- estimated start_time and end_time in seconds (within {video_duration_sec}s total)

Output in JSON format:
[
  {{
    "title": "...",
    "description": "...",
    "start_time": 0,
    "end_time": 90
  }},
  ...
]

Transcript:
{transcript}
"""
    try:
        return json.loads(get_completion(prompt))
    except Exception as e:
        st.error("❌ GPT failed to return valid JSON for segments.")
        st.text(str(e))
        return []

# Ask GPT to extract key moments from the transcript
def gpt_extract_highlights(transcript, video_duration_sec=600):
    prompt = f"""
List 5 key moments from this transcript. For each, include:
- a key quote or sentence
- estimated timestamp (0 to {video_duration_sec} seconds)

Output in JSON:
[
  {{
    "text": "...",
    "timestamp": ...
  }},
  ...
]

Transcript:
{transcript}
"""
    try:
        return json.loads(get_completion(prompt))
    except Exception as e:
        st.error("❌ GPT failed to return valid JSON for highlights.")
        st.text(str(e))
        return []

# Generate caption from image using BLIP
def caption_frame(img_path, processor, model):
    image = Image.open(img_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Use DeepFace to detect facial emotion
def detect_emotion(img_path):
    try:
        faces = DeepFace.extract_faces(img_path=img_path, detector_backend='opencv', enforce_detection=False)
        if len(faces) == 0:
            return None
        else:
            face_results = []
            all_emotions = []
            for face_info in faces:
                face_img = face_info["face"]
                result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                if isinstance(result, list): result = result[0]
                emotions = result["emotion"]
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]
                face_results.append(f"{dominant} ({round(confidence, 2)}%)")
                all_emotions.append(emotions)
            return face_results, all_emotions
    except Exception as e:
        return f"Error: {e}", []


# Sample frames from the video at regular intervals
def sample_frames(video_path, interval_sec=30, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    frames = []
    count = 0
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_path = f"{output_dir}/frame_{count:03d}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append((frame_path, int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)))
            count += 1
        idx += 1
    cap.release()
    return frames

def format_timestamp(seconds):
    mins, secs = divmod(seconds, 60)
    return f"{int(mins):02}:{int(secs):02}"

def inject_frame_descriptions(transcript_segments, frame_descriptions):
    enriched_lines = []
    i = 0  # frame index

    for seg in transcript_segments:
        seg_time = seg['start']
        seg_text = seg['text'].strip()

        while i < len(frame_descriptions) and frame_descriptions[i][0] <= seg_time:
            desc = frame_descriptions[i][1]
            enriched_lines.append(f"[Visual context @ {format_timestamp(frame_descriptions[i][0])}]: {desc}")
            i += 1

        if seg_text:
            enriched_lines.append(f"[{format_timestamp(seg['start'])}] {seg_text}")

    while i < len(frame_descriptions):
        desc = frame_descriptions[i][1]
        enriched_lines.append(f"[Visual context @ {format_timestamp(frame_descriptions[i][0])}]: {desc}")
        i += 1

    return "\n".join(enriched_lines)

def get_nearest_dialogue_snippet(transcript, timestamp, window=15):
    """Extract lines within ±window seconds of the given timestamp"""
    lines = nltk.sent_tokenize(transcript)
    segment_duration = max(1, len(lines) // max(1, st.session_state.duration // window))
    idx = int(timestamp / window)
    start = max(0, idx - 1)
    end = min(len(lines), idx + 2)
    return " ".join(lines[start:end])



# Start of Streamlit UI
st.image('BriefTube_streamlit.png')
st.title("BriefTube − A YouTube Video Summarizer")

video_url = st.text_input("Paste a YouTube video URL:")

# Transcribing video
if st.button("Transcribe & Analyze Full Video"):
    if not video_url:
        st.warning("Please enter a valid URL.")
    else:
        try:
          st.info('Getting video metadata..')
          result = subprocess.run(
          ["yt-dlp", "-j", video_url], capture_output=True, text=True, check=True
          )
          video_info = json.loads(result.stdout)
          duration_sec = video_info.get("duration", 600)
          st.session_state.duration = duration_sec
        except Exception as e:
          st.warning("Could not fetch video duration. Using default value.")
          duration_sec = 600
          st.session_state.duration = duration_sec

        try:
            video_id = str(uuid.uuid4())[:8]
            audio_path = f"{video_id}.mp3"
            video_path = f"{video_id}.mp4"
            duration_sec = 600  # Default estimate

            # Download video and audio with yt-dlp using URL connection
            st.info('Downloading video...')
            subprocess.run(["yt-dlp", "-f", "mp4", "-o", video_path, video_url], check=True)
            subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, video_url], check=True)

            # Transcribe audio with Whisper model
            st.info("Transcribing audio with Whisper...")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            transcript = result["text"]


            st.session_state.full_transcript = transcript
            st.session_state.frames = sample_frames(video_path)
            st.session_state.video_path = video_path
            st.session_state.audio_path = audio_path

            st.subheader("Full Transcript Preview")
            st.write(transcript[:1000] + "...")

        except Exception as e:
            st.error(f"Error: {e}")

# Letting user trim the transcript and summarize only that portion
if "full_transcript" in st.session_state:
    st.subheader("Trim Transcript for Summary")
    selected_text = st.text_area("You can trim or highlight a part of the transcript:",
                                 value=st.session_state.full_transcript,
                                 height=300,
                                 key="trimmed_text")

    if st.button("Summarize Selected Portion"):
        st.info("Summarizing trimmed selection...")

        # Optional: Convert to list of segments (assuming 1 per sentence or ~30s chunks)
        trimmed_lines = nltk.sent_tokenize(selected_text)
        transcript_segments = [
            {"start": i * 30, "text": line} for i, line in enumerate(trimmed_lines)
        ]

        # Build frame_descriptions from previously sampled frames
        processor, blip_model = load_blip_model()
        frame_descriptions = []
        for frame_path, timestamp in st.session_state.frames:
            caption = caption_frame(frame_path, processor, blip_model)
            frame_descriptions.append((timestamp, caption))

        enriched_transcript = inject_frame_descriptions(transcript_segments, frame_descriptions)
        #st.write(enriched_transcript)

        # Summarizing the final transcript, sending to GPT
        summary = get_completion(f"Summarize this enriched transcript with visual context:\n\n{enriched_transcript}")
        st.subheader("Summary of Selected Portion")
        st.write(summary)


# Step 3: Proceed with full video analysis (segments, highlights, visual metadata)
if "full_transcript" in st.session_state and st.button("Proceed with Full Video Analysis"):
    st.info("GPT topic segmentation (full video)...")
    segments = gpt_segment_transcript(st.session_state.full_transcript, st.session_state.duration)
    st.subheader("Topic Segments")
    for seg in segments:
      start_mins, start_secs = divmod(seg['start_time'], 60)
      end_mins, end_secs = divmod(seg['end_time'], 60)
      st.markdown(f"**{seg['title']}** ([{start_mins}:{start_secs:02}]({video_url}&t={seg['start_time']})–[{end_mins}:{end_secs:02}]({video_url}&t={seg['end_time']}))")
      st.text(seg['description'])

    st.info("GPT highlight extraction (full video)...")
    highlights = gpt_extract_highlights(st.session_state.full_transcript, st.session_state.duration)
    st.subheader("🌟 Key Highlights")
    for h in highlights:
      #st.markdown(f"{h['timestamp']}")
      timestamp = int(float(h['timestamp']))
      mins, secs = divmod(timestamp, 60)

      # mins, secs = divmod(h['timestamp'], 60)
      st.markdown(f"**[{mins}:{secs:02}]({video_url}&t={h['timestamp']})** — {h['text']}")

    st.subheader("🧠 Meta Analysis (Every 30 Seconds)")
    processor, blip_model = load_blip_model()
    if "trimmed_text" in st.session_state and st.session_state.trimmed_text.strip():
      dialogue_source = st.session_state.trimmed_text
    else:
      dialogue_source = st.session_state.full_transcript

# Analyze each frame
    for frame_path, timestamp in st.session_state.frames:
        mins, secs = divmod(timestamp, 60)
        st.markdown(f"### ⏱️ [{mins}:{secs:02}]({video_url}&t={timestamp})")

        # Show frame
        st.image(frame_path, use_container_width=True)

        # Generate caption
        caption = caption_frame(frame_path, processor, blip_model)
        st.markdown(f"**🖼️ Scene Description:** *{caption}*")

        # Extract matching transcript snippet
        dialogue_snippet = get_nearest_dialogue_snippet(dialogue_source, timestamp)

        # Enriched GPT prompt
        meta_prompt = f"""
        Given the following visual caption and dialogue context, infer scene metadata.
    
        Caption: "{caption}"
    
        Transcript/Dialogue Context:
        "{dialogue_snippet}"
    
        Infer the following:
        - Location
        - Weather
        - Situation
        - Audience
        - Emotion (non-facial)
        """

        metadata_text = get_completion(meta_prompt)
        st.markdown(f"**🧩 Inferred Metadata:**\n```\n{metadata_text}\n```")

        # Run facial emotion detection
        if DEEPFACE_AVAILABLE:
            emotion_labels, full_scores = detect_emotion(frame_path)
            if emotion_labels is not None:
                st.markdown("**😃 Facial Emotion (DeepFace):**")
                for i, label in enumerate(emotion_labels):
                    st.write(f"Face {i+1}: {label}")
                # st.markdown("**😶 Full Emotion Scores:**")
                # for i, scores in enumerate(full_scores):
                    # st.markdown(f"Face {i+1}:")
                    # st.json(scores)


                #else:
                #    st.error(emotion_labels)
            else:
                st.markdown('No faces detected.')
        else:
            st.markdown("*😐 DeepFace not installed.*")

  
    if st.button("Clean Up Files"):
      os.remove(st.session_state.audio_path)
      os.remove(st.session_state.video_path)
      for f, _ in st.session_state.frames:
        os.remove(f)
      st.success("Temporary files deleted.")


! streamlit run app.py & npx localtunnel --port 8501

