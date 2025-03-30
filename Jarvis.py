import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text.lower()
        except:
            return ""

def jarvis_response(command):
    if "hello" in command:
        return "Hello, sir. How can I assist you?"
    elif "time" in command:
        return f"The time is {datetime.datetime.now().strftime('%H:%M')}"
    elif "open youtube" in command:
        webbrowser.open("https://youtube.com")
        return "Opening YouTube."
    elif "exit" in command:
        speak("Goodbye, sir.")
        exit()
    else:
        return "I didn't understand that."

while True:
    command = listen()
    print("You said:", command)
    response = jarvis_response(command)
    print("JARVIS:", response)
    speak(response)
