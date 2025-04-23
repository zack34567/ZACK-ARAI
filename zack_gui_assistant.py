import sys
import os
import time
import io
import pytesseract
import base64
import speech_recognition as sr
from PIL import ImageGrab, ImageFilter, ImageEnhance, Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QListWidget, QLineEdit, QComboBox, QSystemTrayIcon, 
    QMenu, QListWidgetItem, QMessageBox, QInputDialog, QFileDialog, QFrame, QDialog,
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QByteArray, QPropertyAnimation, QRect, QTimer
from PyQt5.QtGui import (QIcon, QColor, QPixmap, QImage, QPainter, 
                         QTextCursor, QFont, QPalette, QLinearGradient, QMovie)
import qdarkstyle
import pyttsx3
from gtts import gTTS
import playsound
from groq import Groq
import requests
import tempfile
import torch
from diffusers import StableDiffusionPipeline
import pyautogui
import webbrowser
import json
import keyboard
import pyperclip

# ========== CONFIGURATION ========== #
GROQ_API_KEY = REPLACE IT WITH YOUR TOKEN 
HUGGINGFACE_TOKEN = REPLACE IT WITH YOUR TOKEN

# Check GPU availability
HAS_GPU = torch.cuda.is_available()
if HAS_GPU:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected - using CPU only")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ========== PREMIUM UTILITY FUNCTIONS ========== #
def speak(text, use_gtts=True):
    """High-quality text-to-speech with multiple fallbacks"""
    try:
        if use_gtts:
            tts = gTTS(text=text[:500], lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=True) as fp:
                temp_path = f"{fp.name}.mp3"
                tts.save(temp_path)
                playsound.playsound(temp_path)
                time.sleep(0.1)
                try: os.remove(temp_path)
                except: pass
        else:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)
            engine.say(text[:500])
            engine.runAndWait()
    except Exception as e:
        print(f"[TTS Error] {e}")
        os.system(f'say "{text[:200]}"' if sys.platform == 'darwin' else f'espeak "{text[:200]}"')

def listen():
    """Professional-grade voice recognition"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    recognizer.dynamic_energy_threshold = True
    
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            print("Speak now...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            print("Processing...")
            query = recognizer.recognize_google(audio)
            print(f"Recognized: {query}")
            return query
        except sr.WaitTimeoutError:
            print("No speech detected")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"API error: {e}")
            return ""

def capture_and_ocr():
    """Industrial-grade OCR with advanced preprocessing"""
    try:
        screenshot = ImageGrab.grab()
        screenshot = screenshot.convert("L")
        enhancer = ImageEnhance.Contrast(screenshot)
        screenshot = enhancer.enhance(2.0)
        screenshot = screenshot.point(lambda x: 0 if x < 140 else 255, '1')
        screenshot = screenshot.filter(ImageFilter.EDGE_ENHANCE_MORE)
        screenshot = screenshot.filter(ImageFilter.MedianFilter(size=3))
        screenshot = screenshot.filter(ImageFilter.SHARPEN)

        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()@\'\" "'
        text = pytesseract.image_to_string(screenshot, config=custom_config)
        return text.strip() if text else "No readable text found."
    except Exception as e:
        return f"OCR Error: {str(e)}"

def image_to_text_summary(image_path):
    """Extract and summarize text from an image"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        
        if not text.strip():
            return "No readable text found in the image."
            
        # Get summary from AI
        prompt = f"Summarize this extracted text in 3-5 bullet points:\n\n{text[:3000]}"
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error processing image: {str(e)}"

# ========== WORKFLOW AUTOMATION ========== #
class WorkflowEngine:
    def __init__(self):
        self.workflows = self.load_workflows()
        self.current_workflow = None
        
    def load_workflows(self):
        """Load predefined workflows from file"""
        try:
            workflow_path = os.path.join(os.path.expanduser("~"), "ZACK_workflows.json")
            if os.path.exists(workflow_path):
                with open(workflow_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading workflows: {e}")
            return {}
            
    def save_workflows(self):
        """Save workflows to file"""
        try:
            workflow_path = os.path.join(os.path.expanduser("~"), "ZACK_workflows.json")
            with open(workflow_path, 'w') as f:
                json.dump(self.workflows, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving workflows: {e}")
            return False
    
    def add_workflow(self, name, steps):
        """Add a new workflow"""
        self.workflows[name.lower()] = steps
        return self.save_workflows()
        
    def execute_workflow(self, workflow_name):
        """Execute a workflow by name"""
        workflow = self.workflows.get(workflow_name.lower())
        if not workflow:
            return False, f"Workflow '{workflow_name}' not found"
            
        self.current_workflow = workflow_name
        try:
            for step in workflow:
                self.execute_step(step)
            return True, f"Workflow '{workflow_name}' completed successfully"
        except Exception as e:
            return False, f"Error executing workflow: {str(e)}"
        finally:
            self.current_workflow = None
            
    def execute_step(self, step):
        """Execute a single workflow step"""
        action = step.get("action")
        params = step.get("params", {})
        
        if action == "type":
            self._type_text(params.get("text", ""), params.get("delay", 0.1))
        elif action == "press":
            self._press_key(params.get("key", ""))
        elif action == "hotkey":
            self._hotkey(params.get("keys", []))
        elif action == "click":
            self._click(params.get("x", 0), params.get("y", 0))
        elif action == "move":
            self._move(params.get("x", 0), params.get("y", 0))
        elif action == "wait":
            self._wait(params.get("seconds", 1))
        elif action == "open":
            self._open_app(params.get("app", ""))
        elif action == "website":
            self._open_website(params.get("url", ""))
        elif action == "copy":
            self._copy_to_clipboard(params.get("text", ""))
        elif action == "paste":
            self._paste_from_clipboard()
        elif action == "speak":
            speak(params.get("text", ""))
            
    def _type_text(self, text, delay=0.1):
        """Type text with optional delay between keystrokes"""
        pyautogui.write(text, interval=delay)
        
    def _press_key(self, key):
        """Press a single key"""
        pyautogui.press(key)
        
    def _hotkey(self, keys):
        """Press a combination of keys"""
        pyautogui.hotkey(*keys)
        
    def _click(self, x, y):
        """Click at specific coordinates"""
        pyautogui.click(x, y)
        
    def _move(self, x, y):
        """Move mouse to specific coordinates"""
        pyautogui.moveTo(x, y)
        
    def _wait(self, seconds):
        """Wait for specified time"""
        time.sleep(seconds)
        
    def _open_app(self, app_name):
        """Open an application"""
        if sys.platform == "win32":
            os.startfile(app_name)
        elif sys.platform == "darwin":
            os.system(f"open -a '{app_name}'")
        else:
            os.system(f"{app_name} &")
            
    def _open_website(self, url):
        """Open a website in default browser"""
        webbrowser.open(url)
        
    def _copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        pyperclip.copy(text)
        
    def _paste_from_clipboard(self):
        """Paste from clipboard"""
        pyautogui.hotkey('ctrl', 'v')

# ========== AI SERVICES ========== #
class GroqAIThread(QThread):
    response_ready = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            self.status_update.emit("🔍 Processing with Groq...")
            messages = [{"role": "user", "content": self.prompt}]
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                stream=False
            )
            response_text = completion.choices[0].message.content
            self.response_ready.emit(response_text)
        except Exception as e:
            self.response_ready.emit(f"[AI Error] {str(e)}")

class LocalAIThread(QThread):
    response_ready = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            self.status_update.emit("🔍 Processing with Local AI...")
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "llama3",
                    "prompt": self.prompt,
                    "stream": False
                },
                timeout=45
            )
            self.response_ready.emit(response.json().get('response', ''))
        except Exception as e:
            self.response_ready.emit(f"[Local AI Error] {str(e)}")

class ImageGenThread(QThread):
    image_ready = pyqtSignal(bytes)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        self.local_pipe = None
        
        if HAS_GPU:
            try:
                self.status_update.emit("⚙️ Loading local model...")
                self.local_pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to("cuda")
                self.local_pipe.enable_attention_slicing()
                self.status_update.emit("✅ Local model ready!")
            except Exception as e:
                self.error_occurred.emit(f"Local model failed: {str(e)}")

    def try_generation(self):
        if HUGGINGFACE_TOKEN:
            self.status_update.emit("🌐 Attempting Hugging Face API...")
            result = self.try_huggingface()
            if result: return result
        
        if HAS_GPU and self.local_pipe:
            self.status_update.emit("⚡ Using local GPU...")
            return self.try_local()
        
        return None

    def try_huggingface(self):
        try:
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
            
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": self.prompt},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.content
            else:
                self.error_occurred.emit(f"API Error: {response.text}")
                return None
                
        except Exception as e:
            self.error_occurred.emit(f"Hugging Face Error: {str(e)}")
            return None

    def try_local(self):
        try:
            image = self.local_pipe(
                self.prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
            
        except Exception as e:
            self.error_occurred.emit(f"Local generation failed: {str(e)}")
            return None

    def run(self):
        result = self.try_generation()
        if result:
            self.image_ready.emit(result)
        else:
            self.error_occurred.emit("❌ All generation methods failed")

# ========== PREMIUM GUI CLASS ========== #
class ZACKChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 ZACK PRO AI Assistant")
        self.setWindowIcon(QIcon("ai_icon.png"))
        self.resize(1280, 800)
        self.conversation_history = []
        self.is_listening = False
        self.workflow_engine = WorkflowEngine()
        
        # Create storage directories
        self.image_storage = os.path.join(os.path.expanduser("~"), "ZACK_AI_Images")
        self.audio_storage = os.path.join(os.path.expanduser("~"), "ZACK_AI_Audio")
        os.makedirs(self.image_storage, exist_ok=True)
        os.makedirs(self.audio_storage, exist_ok=True)
        
        self.init_ui()
        self.apply_premium_styles()
        self.setup_tray()
        self.setup_animations()
        
        # Load thinking animation
        self.thinking_movie = QMovie("thinking.gif")  # Replace with your GIF path
        self.thinking_movie.setScaledSize(QSize(100, 100))
        self.thinking_label = QLabel()
        self.thinking_label.setMovie(self.thinking_movie)
        self.thinking_label.hide()
        self.thinking_label.setAlignment(Qt.AlignCenter)
        self.thinking_label.setFixedSize(100, 100)
        self.thinking_label.setStyleSheet("background: transparent;")

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # Left Sidebar with gradient background
        sidebar = QVBoxLayout()
        sidebar.setSpacing(15)
        
        # Sidebar frame for better visual separation
        sidebar_frame = QFrame()
        sidebar_frame.setFrameShape(QFrame.StyledPanel)
        sidebar_frame.setLayout(sidebar)
        sidebar_frame.setMinimumWidth(280)
        sidebar_frame.setMaximumWidth(300)

        # App Title
        title = QLabel("ZACK PRO AI")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Segoe UI", 18, QFont.Bold)
        title.setFont(title_font)
        sidebar.addWidget(title)

        # Model Selection
        model_group = QVBoxLayout()
        model_label = QLabel("🧠 AI Engine:")
        model_label.setFont(QFont("Segoe UI", 10))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Groq (Online)", "Ollama (Local)"])
        model_group.addWidget(model_label)
        model_group.addWidget(self.model_selector)
        sidebar.addLayout(model_group)

        # Voice Button with modern design
        self.btn_voice = QPushButton("🎤 Voice Command")
        self.btn_voice.setCheckable(True)
        self.btn_voice.setToolTip("Press and hold to speak")
        self.btn_voice.pressed.connect(self.start_listening)
        self.btn_voice.released.connect(self.stop_listening)
        sidebar.addWidget(self.btn_voice)

        # Action Buttons with icons
        self.btn_capture = QPushButton("📸 Screen OCR")
        self.btn_capture.clicked.connect(self.capture_screen)
        sidebar.addWidget(self.btn_capture)

        self.btn_imggen = QPushButton("🎨 Generate Image")
        self.btn_imggen.clicked.connect(self.generate_image)
        sidebar.addWidget(self.btn_imggen)
        
        # New Workflow Buttons
        self.btn_workflow = QPushButton("⚙️ Workflows")
        self.btn_workflow.clicked.connect(self.manage_workflows)
        sidebar.addWidget(self.btn_workflow)
        
        self.btn_imgtext = QPushButton("📝 Image to Text")
        self.btn_imgtext.clicked.connect(self.process_image_text)
        sidebar.addWidget(self.btn_imgtext)

        self.btn_clear = QPushButton("🧹 Clear Chat")
        self.btn_clear.clicked.connect(self.clear_chat)
        sidebar.addWidget(self.btn_clear)

        # Status Indicators
        status_group = QVBoxLayout()
        gpu_status = QLabel(f"🖥️ GPU: {'Enabled' if HAS_GPU else 'Disabled'}")
        gpu_status.setToolTip(f"Device: {torch.cuda.get_device_name(0) if HAS_GPU else 'CPU only'}")
        status_group.addWidget(gpu_status)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 9))
        status_group.addWidget(self.status_label)
        
        sidebar.addLayout(status_group)
        sidebar.addStretch()

        # Main Content Area
        right_panel = QVBoxLayout()
        
        # Chat Display with modern styling
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        right_panel.addWidget(self.chat_display)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                color: #e0e0e0;
            }
        """)
        
        # Input Area with send button
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_layout = QHBoxLayout(input_frame)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask ZACK anything or press 🎤 for voice...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.btn_send = QPushButton("🚀 Send")
        self.btn_send.clicked.connect(self.send_message)
        input_layout.addWidget(self.btn_send)

        right_panel.addWidget(input_frame)

        # Add widgets to main layout
        main_layout.addWidget(sidebar_frame)
        main_layout.addLayout(right_panel)

    def apply_premium_styles(self):
        """Premium styling with gradients and animations"""
        # Base style
        self.setStyleSheet("""
            QWidget {
                background: #1e1e2e;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
            
            QFrame {
                background: rgba(30, 30, 46, 0.8);
                border-radius: 10px;
                border: 1px solid #313244;
            }
            
            QTextEdit {
                background: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                color: #e0e0e0;
                selection-background-color: #89b4fa;
            }
            
            QLineEdit {
                background: #2e2e3e;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                color: white;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #89b4fa, stop:1 #74c7ec);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                min-width: 120px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #94e2d5, stop:1 #89dceb);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #74c7ec, stop:1 #89b4fa);
            }
            
            #btn_voice {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #f38ba8, stop:1 #eba0ac);
            }
            
            #btn_voice:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #d20f39, stop:1 #e64553);
            }
            
            QComboBox {
                background: #2e2e3e;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 8px;
                min-width: 120px;
            }
            
            QComboBox::drop-down {
                border: none;
            }
        """)

        # Set custom fonts
        font = QFont("Segoe UI", 12)
        self.setFont(font)
        
        # Chat display font
        chat_font = QFont("Segoe UI", 12)
        self.chat_display.setFont(chat_font)

    def setup_animations(self):
        """Setup UI animations"""
        # Message fade animation
        self.message_animation = QPropertyAnimation(self.chat_display.viewport(), b"opacity")
        self.message_animation.setDuration(500)
        self.message_animation.setStartValue(0)
        self.message_animation.setEndValue(1)
        
        # Button pulse animation
        self.btn_animation = QPropertyAnimation(self.btn_send, b"geometry")
        self.btn_animation.setDuration(300)
        self.btn_animation.setStartValue(QRect(0, 0, 80, 40))
        self.btn_animation.setEndValue(QRect(0, 0, 85, 45))
        self.btn_animation.setLoopCount(2)
        self.btn_animation.setDirection(QPropertyAnimation.Backward)

    def setup_tray(self):
        """Modern system tray integration"""
        self.tray_icon = QSystemTrayIcon(QIcon("ai_icon.png"), self)
        self.tray_icon.setToolTip("ZACK PRO AI Assistant")
        
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show Window")
        quit_action = tray_menu.addAction("Exit")
        
        show_action.triggered.connect(self.show_normal)
        quit_action.triggered.connect(QApplication.quit)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def show_normal(self):
        """Restore and bring window to front"""
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        self.raise_()
        self.activateWindow()

    def append_to_chat(self, sender, message):
        """Premium chat message formatting"""
        if sender.lower() == "you":
            msg_html = f"""
            <div style='margin: 10px; text-align: right;'>
                <span style='color: #94e2d5; font-weight: bold; font-size: 14px;'>{sender}</span>
                <div style='background: #313244; padding: 12px; border-radius: 12px; 
                            display: inline-block; max-width: 80%; margin-top: 5px;
                            border: 1px solid #45475a;'>
                    {message}
                </div>
            </div>
            """
        else:
            msg_html = f"""
            <div style='margin: 10px;'>
                <span style='color: #89b4fa; font-weight: bold; font-size: 14px;'>{sender}</span>
                <div style='background: #1e1e2e; padding: 12px; border-radius: 12px;
                            display: inline-block; max-width: 80%; margin-top: 5px;
                            border: 1px solid #313244;'>
                    {message}
                </div>
            </div>
            """
        
        self.chat_display.append(msg_html)
        self.chat_display.moveCursor(QTextCursor.End)
        self.message_animation.start()
        self.conversation_history.append({"role": sender, "content": message})

    def update_status(self, message):
        """Animated status updates"""
        self.status_label.setText(message)
        QTimer.singleShot(3000, lambda: self.status_label.setText("Ready"))

    def start_listening(self):
        """Voice listening with visual feedback"""
        self.is_listening = True
        self.btn_voice.setChecked(True)
        self.append_to_chat("System", "Listening... (release button when done)")
        self.update_status("🎤 Listening...")

    def stop_listening(self):
        """Process voice input"""
        self.btn_voice.setChecked(False)
        if not self.is_listening:
            return
            
        self.is_listening = False
        self.update_status("Processing voice...")
        
        self.voice_thread = QThread()
        self.voice_worker = VoiceWorker()
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_worker.finished.connect(self.handle_voice_result)
        self.voice_worker.finished.connect(self.voice_thread.quit)
        self.voice_worker.finished.connect(self.voice_worker.deleteLater)
        self.voice_thread.finished.connect(self.voice_thread.deleteLater)
        self.voice_thread.start()

    def handle_voice_result(self, query):
        if query:
            self.input_field.setText(query)
            self.btn_animation.start()  # Visual feedback
            self.send_message()
        else:
            self.append_to_chat("System", "Voice input failed")

    def send_message(self):
        prompt = self.input_field.text().strip()
        if not prompt:
            return

        self.append_to_chat("You", prompt)
        self.input_field.clear()

        # Check if this is a workflow command
        if prompt.lower().startswith("run workflow"):
            workflow_name = prompt[12:].strip()
            self.execute_workflow(workflow_name)
            return
            
        # Show thinking animation
        self.show_thinking_animation()
        
        if self.model_selector.currentText() == "Groq (Online)":
            self.thread = GroqAIThread(prompt)
        else:
            self.thread = LocalAIThread(prompt)

        self.thread.response_ready.connect(self.handle_response)
        self.thread.status_update.connect(self.update_status)
        self.thread.finished.connect(self.hide_thinking_animation)
        self.thread.start()
        self.update_status("Processing request...")

    def show_thinking_animation(self):
        """Show thinking animation GIF"""
        self.thinking_label.show()
        self.thinking_movie.start()

    def hide_thinking_animation(self):
        """Hide thinking animation GIF"""
        self.thinking_movie.stop()
        self.thinking_label.hide()

    def handle_response(self, response):
        self.append_to_chat("ZACK", response)
        speak(response, use_gtts=True)
        self.update_status("Response received")

    def capture_screen(self):
        self.update_status("Capturing screen...")
        self.append_to_chat("System", "Performing screen OCR...")
        
        self.ocr_thread = QThread()
        self.ocr_worker = OCRWorker()
        self.ocr_worker.moveToThread(self.ocr_thread)
        self.ocr_thread.started.connect(self.ocr_worker.run)
        self.ocr_worker.finished.connect(self.handle_ocr_result)
        self.ocr_worker.finished.connect(self.ocr_thread.quit)
        self.ocr_worker.finished.connect(self.ocr_worker.deleteLater)
        self.ocr_thread.finished.connect(self.ocr_thread.deleteLater)
        self.ocr_thread.start()

    def handle_ocr_result(self, text):
        if text:
            self.append_to_chat("OCR Result", text)
            self.input_field.setText(f"Analyze this text: {text[:200]}...")
        else:
            self.append_to_chat("System", "OCR failed or no text found")
        self.update_status("OCR completed")

    def generate_image(self):
        prompt, ok = QInputDialog.getText(self, "🎨 Generate Image", "Describe the image you want:")
        if ok and prompt:
            self.append_to_chat("You", f"[Image Request] {prompt}")
            
            self.img_thread = ImageGenThread(prompt)
            self.img_thread.image_ready.connect(self.display_image)
            self.img_thread.error_occurred.connect(self.handle_image_error)
            self.img_thread.status_update.connect(self.update_status)
            self.img_thread.start()
            self.update_status("Starting image generation...")

    def handle_image_error(self, error):
        self.append_to_chat("System", f"❌ {error}")
        QMessageBox.critical(self, "Image Generation Error", error)
        self.update_status("Image generation failed")

    def display_image(self, image_bytes):
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_path = os.path.join(self.image_storage, f"image_{timestamp}.png")
            with open(img_path, 'wb') as f:
                f.write(image_bytes)
            
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            html = f"""
            <div style='margin: 15px; border: 1px solid #45475a; border-radius: 10px; padding: 10px;'>
                <img src='data:image/png;base64,{base64_image}' 
                     style='max-width: 100%; max-height: 400px; display: block; margin: 0 auto;
                            border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'/>
                <p style='color: #89b4fa; text-align: center; margin-top: 8px;'>Image saved to: {img_path}</p>
            </div>
            """
            self.chat_display.append(html)
            self.chat_display.moveCursor(QTextCursor.End)
            self.update_status("Image generated successfully")
            
        except Exception as e:
            self.handle_image_error(f"Failed to display image: {str(e)}")

    def process_image_text(self):
        """Process image to extract and summarize text"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.append_to_chat("System", f"Processing image: {os.path.basename(file_path)}")
            self.update_status("Extracting text from image...")
            
            self.imgtext_thread = QThread()
            self.imgtext_worker = ImageTextWorker(file_path)
            self.imgtext_worker.moveToThread(self.imgtext_thread)
            self.imgtext_thread.started.connect(self.imgtext_worker.run)
            self.imgtext_worker.finished.connect(self.handle_imgtext_result)
            self.imgtext_worker.finished.connect(self.imgtext_thread.quit)
            self.imgtext_worker.finished.connect(self.imgtext_worker.deleteLater)
            self.imgtext_thread.finished.connect(self.imgtext_thread.deleteLater)
            self.imgtext_thread.start()

    def handle_imgtext_result(self, result):
        if result:
            self.append_to_chat("ZACK", f"📝 Image Text Summary:\n\n{result}")
            speak("Here's the summary of the image text", use_gtts=True)
        else:
            self.append_to_chat("System", "Failed to process image text")
        self.update_status("Image processing completed")

    def manage_workflows(self):
        """Manage workflows dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("⚙️ Workflow Management")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Workflow list
        self.workflow_list = QListWidget()
        self.workflow_list.addItems(self.workflow_engine.workflows.keys())
        layout.addWidget(self.workflow_list)
        
        # Button row
        btn_layout = QHBoxLayout()
        
        btn_add = QPushButton("➕ Add")
        btn_add.clicked.connect(lambda: self.add_workflow(dialog))
        btn_layout.addWidget(btn_add)
        
        btn_edit = QPushButton("✏️ Edit")
        btn_edit.clicked.connect(lambda: self.edit_workflow(dialog))
        btn_layout.addWidget(btn_edit)
        
        btn_run = QPushButton("▶️ Run")
        btn_run.clicked.connect(lambda: self.run_selected_workflow(dialog))
        btn_layout.addWidget(btn_run)
        
        btn_delete = QPushButton("🗑️ Delete")
        btn_delete.clicked.connect(self.delete_workflow)
        btn_layout.addWidget(btn_delete)
        
        layout.addLayout(btn_layout)
        
        dialog.exec_()

    def add_workflow(self, parent_dialog):
        """Add a new workflow"""
        name, ok = QInputDialog.getText(
            parent_dialog, "Add Workflow", "Workflow name:"
        )
        if ok and name:
            # Get workflow steps from user
            steps, ok = QInputDialog.getMultiLineText(
                parent_dialog, "Workflow Steps", 
                "Enter steps as JSON (list of {action, params} objects):",
                '[{"action": "type", "params": {"text": "Hello", "delay": 0.1}}]'
            )
            
            if ok and steps:
                try:
                    steps_json = json.loads(steps)
                    if self.workflow_engine.add_workflow(name, steps_json):
                        self.workflow_list.addItem(name)
                        QMessageBox.information(
                            parent_dialog, "Success", 
                            f"Workflow '{name}' added successfully!"
                        )
                    else:
                        QMessageBox.warning(
                            parent_dialog, "Error", 
                            "Failed to save workflow"
                        )
                except json.JSONDecodeError:
                    QMessageBox.critical(
                        parent_dialog, "Invalid JSON", 
                        "Please enter valid JSON for workflow steps"
                    )

    def edit_workflow(self, parent_dialog):
        """Edit an existing workflow"""
        selected = self.workflow_list.currentItem()
        if not selected:
            QMessageBox.warning(parent_dialog, "No Selection", "Please select a workflow to edit")
            return
            
        name = selected.text()
        workflow = self.workflow_engine.workflows.get(name)
        if not workflow:
            return
            
        # Show current steps for editing
        steps, ok = QInputDialog.getMultiLineText(
            parent_dialog, "Edit Workflow", 
            "Edit workflow steps as JSON:",
            json.dumps(workflow, indent=2)
        )
        
        if ok and steps:
            try:
                steps_json = json.loads(steps)
                self.workflow_engine.workflows[name] = steps_json
                if self.workflow_engine.save_workflows():
                    QMessageBox.information(
                        parent_dialog, "Success", 
                        f"Workflow '{name}' updated successfully!"
                    )
                else:
                    QMessageBox.warning(
                        parent_dialog, "Error", 
                        "Failed to save workflow changes"
                    )
            except json.JSONDecodeError:
                QMessageBox.critical(
                    parent_dialog, "Invalid JSON", 
                    "Please enter valid JSON for workflow steps"
                )

    def run_selected_workflow(self, parent_dialog):
        """Run the selected workflow"""
        selected = self.workflow_list.currentItem()
        if not selected:
            QMessageBox.warning(parent_dialog, "No Selection", "Please select a workflow to run")
            return
            
        name = selected.text()
        self.execute_workflow(name)

    def execute_workflow(self, workflow_name):
        """Execute a workflow"""
        self.append_to_chat("System", f"Executing workflow: {workflow_name}")
        self.update_status(f"Running workflow: {workflow_name}")
        
        # Run in a separate thread to keep UI responsive
        self.workflow_thread = QThread()
        self.workflow_worker = WorkflowWorker(self.workflow_engine, workflow_name)
        self.workflow_worker.moveToThread(self.workflow_thread)
        self.workflow_thread.started.connect(self.workflow_worker.run)
        self.workflow_worker.finished.connect(self.handle_workflow_result)
        self.workflow_worker.finished.connect(self.workflow_thread.quit)
        self.workflow_worker.finished.connect(self.workflow_worker.deleteLater)
        self.workflow_thread.finished.connect(self.workflow_thread.deleteLater)
        self.workflow_thread.start()

    def handle_workflow_result(self, success, message):
        if success:
            self.append_to_chat("System", f"✅ {message}")
            speak("Workflow completed successfully", use_gtts=True)
        else:
            self.append_to_chat("System", f"❌ {message}")
            speak("Workflow failed", use_gtts=True)
        self.update_status(message)

    def delete_workflow(self):
        """Delete the selected workflow"""
        selected = self.workflow_list.currentItem()
        if not selected:
            return
            
        name = selected.text()
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete workflow '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.workflow_engine.workflows[name]
            if self.workflow_engine.save_workflows():
                self.workflow_list.takeItem(self.workflow_list.row(selected))
                QMessageBox.information(
                    self, "Success", 
                    f"Workflow '{name}' deleted successfully!"
                )
            else:
                QMessageBox.warning(
                    self, "Error", 
                    "Failed to delete workflow"
                )

    def clear_chat(self):
        self.chat_display.clear()
        self.conversation_history = []
        self.update_status("Chat cleared")

    def closeEvent(self, event):
        self.tray_icon.hide()
        event.accept()

# ========== WORKER CLASSES ========== #
class VoiceWorker(QThread):
    finished = pyqtSignal(str)

    def run(self):
        self.finished.emit(listen())

class OCRWorker(QThread):
    finished = pyqtSignal(str)

    def run(self):
        self.finished.emit(capture_and_ocr())

class ImageTextWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        self.finished.emit(image_to_text_summary(self.image_path))

class WorkflowWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, workflow_engine, workflow_name):
        super().__init__()
        self.workflow_engine = workflow_engine
        self.workflow_name = workflow_name

    def run(self):
        success, message = self.workflow_engine.execute_workflow(self.workflow_name)
        self.finished.emit(success, message)

# ========== RUN APPLICATION ========== #
if __name__ == "__main__":
    # Check for required modules
    try:
        import pytesseract
    except ImportError:
        print("Please install pytesseract: pip install pytesseract")
        sys.exit(1)
        
    try:
        import speech_recognition
    except ImportError:
        print("Please install SpeechRecognition: pip install SpeechRecognition pyaudio")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        window = ZACKChatApp()
        window.show()
        sys.exit(app.exec_())
    else:
        QMessageBox.critical(None, "Invalid API Key", 
            "Please check your Groq API key!\nGet one from: https://console.groq.com/")
        sys.exit(1)
