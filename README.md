# ZACK-GUI-ASSISTANT
🤖 ZACK AR AI Assistant
ZACK AR AI Assistant is a powerful, modern, and modular desktop AI assistant built with Python and PyQt5. Designed to deliver seamless conversations, voice interaction, on-screen text extraction, and integration with advanced AI models (both local and cloud-based), ZACK is your personal AI that fits right into your workflow.

✨ Features
🔹 Dual AI Modes
Choose between:

Groq Cloud AI: Lightning-fast and accurate responses via LLaMA 3 and other powerful models.

Ollama Local AI: Run local models securely without internet access (supports LLaMA-based models).

🔹 Modern GUI Interface

Built with PyQt5 and styled with QDarkStyle

Responsive layout with user & assistant chat bubbles

System tray integration with quick access

🔹 OCR Screen Capture

Instantly capture any part of the screen

Extract text using Tesseract OCR

Useful for code, articles, and problem-solving

🔹 Text-to-Speech (TTS)

Natural voice output using pyttsx3

Read out AI responses for accessibility and multitasking

🔹 Secure API Integration

Uses the Groq SDK for blazing-fast inference

Easy API key configuration

🚀 Tech Stack
Python 3.10+

PyQt5 – GUI

Groq SDK – Cloud AI (LLaMA3)

Ollama – Local AI inference

pytesseract – OCR

pyttsx3 – Text-to-Speech

qdarkstyle – Dark Theme Styling

🔧 How It Works
Select the AI model (Online via Groq or Offline via Ollama)

Type your query or press 📸 to capture screen text

Get intelligent replies with voice output

Everything happens in a sleek, user-friendly GUI

📦 Installation
bash
Copy
Edit
git clone https://github.com/your-username/zack-ar-ai-assistant.git
cd zack-ar-ai-assistant
pip install -r requirements.txt
python app.py
Make sure you:

Install Tesseract OCR

Configure GROQ_API_KEY in the script

Install Ollama if you want to use local models

📌 To-Do (Upcoming Features)
Voice command input 🎙️

Workflow automation 🤖

Remote access via mobile app 📱

Plugin system for developer extensions ⚙

🤝 Contributing
Pull requests are welcome! If you have ideas or want to collaborate on future features, feel free to fork and contribute.

⚡ Credits
Developed by [ABHIJEET]
With ❤️ using Python, PyQt, and Groq API.
