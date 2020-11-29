from skils.skils import search_command
from speech.speech_to_text import creat_text
from speech.text_to_speech import create_voice
# from speech.speaker_identification import identification


print('Говори!')
while True:
    text = creat_text()
    if text != '':
        # if identification() != 'Unknown user':
        answer = search_command(text)
        create_voice(answer)
        print(answer)