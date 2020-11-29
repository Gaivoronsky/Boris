import pyttsx3
engine = pyttsx3.init()


def create_voice(text):
    engine.say(text)
    engine.setProperty('rate', 120)
    engine.setProperty('volume', 0.9)
    engine.runAndWait()
    return