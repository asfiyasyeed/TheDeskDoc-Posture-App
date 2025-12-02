# TheDeskDoc: Automated Body Posture Recognition System 🧘

## Project Overview
TheDeskDoc is an AI-powered, real-time web application designed to help users maintain ergonomic sitting posture. Using a standard webcam, the system analyzes the user's body landmarks to detect spinal slouching and forward head leaning. It provides immediate, corrective feedback to prevent long-term musculoskeletal issues associated with sedentary work.

## Features
* **Real-Time Detection:** Uses Google's MediaPipe framework for high-speed pose estimation.
* **Intelligent Feedback:** Detects specific faults like "Slouching" or "Leaning" using a geometric algorithm.
* **Interactive UI:** Built with **Streamlit**, featuring a clean dashboard and calibration sliders.
* **Timed Alerts:** Only alerts the user if bad posture is maintained for 5 seconds (to avoid false alarms).

## Technologies Used
* **Python:** Core programming language.
* **Streamlit:** For the web interface.
* **MediaPipe:** For human pose estimation.
* **OpenCV:** For image processing.

---

## 🚀 How to Run this Project

This application is built using Streamlit and requires Python. Follow these steps to run it on your local machine using the Command Prompt.

### Step 1: Download the Code
1.  Download this repository (Code -> Download ZIP) and extract it.
2.  Open your **Command Prompt**.
3.  Navigate to the folder where you extracted the files:
    ```cmd
    cd path\to\TheDeskDoc-Posture-App
    ```

### Step 2: Set up the Environment
It is recommended to run this app in a virtual environment to avoid conflicts.

1.  **Create a virtual environment** (run this in your Command Prompt):
    ```cmd
    python -m venv venv
    ```

2.  **Activate the virtual environment**:
    ```cmd
    venv\Scripts\activate
    ```
    *(You should see `(venv)` appear at the start of your command line).*

3.  **Install the required libraries**:
    ```cmd
    pip install -r requirements.txt
    ```

### Step 3: Run the App
Once the installation is complete and your virtual environment is active, run the Streamlit app:

```cmd
streamlit run app.py
