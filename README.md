# 🏏 AI-Powered Cricket Analytics

CoverDriveAI is a computer vision–based cricket shot analysis tool that evaluates **cover drive technique** from video.  
It uses **MediaPipe Pose Estimation** for player movement analysis, **bat path detection** for swing tracking, and a **rule-based grading system** to rate performance as **Beginner**, **Intermediate**, or **Advanced**.  
The project includes a **Streamlit web app** for uploading videos or analyzing YouTube cricket clips.

---

## 🚀 Features
- 📹 **Video Input Options**  
  - Upload a local cricket video file  
  - Paste a YouTube video link (automatic download)
- 🎯 **Pose & Movement Analysis**  
  - Detects player body landmarks using MediaPipe
  - Tracks footwork, head position, swing control, balance, and follow-through
- 🏏 **Bat Path Visualization**  
  - Detects bat movement and creates a swing path plot (`bat_path.png`)
- 📊 **Skill Grade Prediction**  
  - Beginner / Intermediate / Advanced classification
- 🖥 **Streamlit Web Interface**  
  - Watch annotated video preview in the browser
  - Download video, JSON metrics, and PDF report



3.**Install dependencies
## 📂 Project Structure
