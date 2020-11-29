from resemblyzer import preprocess_wav, VoiceEncoder
import os

encoder = VoiceEncoder("cpu")


def identification():
    wav_fpath = 'speech/output.wav'
    wav = preprocess_wav(wav_fpath)

    path_slepok = 'speech/slepok/'
    speaker_names = []
    speaker_wavs = []
    for name in os.listdir(path_slepok):
        path = path_slepok + name
        wav_voice = preprocess_wav(path)
        speaker_wavs.append(wav_voice)
        speaker_names.append(name.replace('.wav', ''))

    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in
                       zip(speaker_names, speaker_embeds)}

    for sample in similarity_dict:
        similarity_dict[sample] = similarity_dict[sample].mean()
    user = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)[0]
    if user[1] > 0.7:
        return user[0]
    return 'Unknown user'