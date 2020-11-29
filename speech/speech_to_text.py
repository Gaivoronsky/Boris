from vosk import Model, KaldiRecognizer, SetLogLevel
import pyaudio
import json
import wave

SetLogLevel(-1)
model = Model("speech/model/kaldi_vosk")


def parse_json(data):
    data = json.loads(data)
    text = ''
    for sample in data['result']:
        text += sample['word'] + ' '
    return text


def creat_text(expectation=9):
    rec = KaldiRecognizer(model, 16000)
    RATE = 16000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    WAVE_OUTPUT_FILENAME = "speech/output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=8000)
    stream.start_stream()
    stop = 0
    text = []
    frames = []

    while True:
        data = stream.read(4000)
        frames.append(data)
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text.append(res['text'])
            print(res['text'])
            stop = 0
            continue
        if json.loads(rec.PartialResult())['partial'] == '':
            stop += 1
        if stop > expectation:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return ' '.join(text)


if __name__ == '__main__':
    creat_text()
    from speaker_identification import identification
    identification()