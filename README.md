# Action Recognition – Human Presence & Activity Detection

This repository contains real-time action recognition code developed by **Irfan Haider** using Python. The model detects human presence and identifies key actions (e.g., tool usage, idle state) for collaborative robotics and smart environments.

---

## 🧠 Features
- YOLOv11-based real-time detection
- Custom action classes (e.g., `grabbing_tool`, `tightening_screw`)
- Basler camera compatibility
- Easy integration with wrist pose estimation

---

## 📁 Project Structure

```bash
action_recognition/
├── model_download.py        # Downloads and loads action recognition model
├── .gitattributes           # Force file type icons on GitHub


🚀 Usage
python model_download.py

