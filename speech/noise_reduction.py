from rnnoise_wrapper import RNNoise

denoiser = RNNoise()

audio = denoiser.read_wav('test.wav')
denoised_audio = denoiser.filter(audio)
denoiser.write_wav('test_denoised.wav', denoised_audio)