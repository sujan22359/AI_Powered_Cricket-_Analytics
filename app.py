import streamlit as st
import os
from cover_drive_analysis_realtime import analyze_video, download_youtube_video

st.set_page_config(page_title="AthleteRise Cricket Analysis", layout="centered")
st.title("🏏 AthleteRise – Cricket Cover Drive Analysis")

# Create temp & output folders
os.makedirs("tmp", exist_ok=True)
os.makedirs("output", exist_ok=True)

# --- Input selection ---
st.subheader("1️⃣ Provide your cricket video")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
youtube_link = st.text_input("Or paste a YouTube video link")

process_btn = st.button("🚀 Analyze Video")

if process_btn:
    if uploaded_file is None and not youtube_link.strip():
        st.error("Please upload a file or provide a YouTube link.")
    else:
        if uploaded_file:
            # Save uploaded file to temp
            video_path = os.path.join("input_video", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.info(f"Using uploaded file: {uploaded_file.name}")
        else:
            st.info("Downloading video from YouTube...")
            try:
                video_path = download_youtube_video(youtube_link)
                st.success("Download complete ✅")
            except Exception as e:
                st.error(f"Failed to download video: {e}")
                st.stop()

        # --- Run analysis ---
        with st.spinner("Analyzing video... this may take a moment"):
            try:
                results = analyze_video(video_path)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        st.success("✅ Analysis complete!")

        # --- Display results ---
        st.subheader("📊 Skill Grade")
        st.markdown(f"**{results['skill_grade']}**")

        st.subheader("📈 Detailed Evaluation")
        st.json(results["evaluation"])

        # --- Show processed video ---
        annotated_video_path = "output/annotated_video.mp4"
        if os.path.exists(annotated_video_path):
            st.subheader("🎥 Annotated Video Preview")
            st.video(annotated_video_path)

        # --- Download buttons ---
        if os.path.exists(annotated_video_path):
            st.download_button(
                "⬇️ Download Annotated Video",
                data=open(annotated_video_path, "rb"),
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )

        json_path = "output/evaluation.json"
        if os.path.exists(json_path):
            st.download_button(
                "⬇️ Download Evaluation JSON",
                data=open(json_path, "rb"),
                file_name="evaluation.json",
                mime="application/json"
            )

        pdf_path = "output/report.pdf"
        if os.path.exists(pdf_path):
            st.download_button(
                "⬇️ Download PDF Report",
                data=open(pdf_path, "rb"),
                file_name="report.pdf",
                mime="application/pdf"
            )
