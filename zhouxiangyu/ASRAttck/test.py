from ASRAdversarialAttacks.AdversarialAttacks import ASRAttacks
import torchaudio
import torch
import numpy as np


bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print(bundle.get_labels())
"""model = bundle.get_model()
torch.save(model, 'D:/1/wav2vec.pkl') """
model = torch.load('D:/1/wav2vec.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#input_audio, sample_rate = torchaudio.load('CRDNN_Model/AudioSamplesASR/spk1_snt1.wav')
input_audio, sample_rate = torchaudio.load('CRDNN_Model/9008/2902-9008-0000.flac')
target_transcription = 'WHEN MAN NOTHING KILL IN BIG CAMEL'
true_transcription = 'THE CHILD ALMOST HURT THE SMALL DOG'
attack = ASRAttacks(model, device, bundle.get_labels())
target = list(target_transcription.upper().replace(" ", "|"))
attack = ASRAttacks(model, device, bundle.get_labels())
temp11 = attack.CW_ATTACK(input_audio, target, epsilon = 0.0015, c = 10,
                  learning_rate = 0.0001, num_iter = 200, decrease_factor_eps = 1,
                  num_iter_decrease_eps = 10, optimizer = None, nested = True,
                  early_stop = True, search_eps = False, targeted = True)

#CW PRINT
print('\n',attack.INFER(torch.from_numpy(temp11)).replace("|"," "))
print(target_transcription)
print("Targeted WER is: ", attack.wer_compute([target_transcription], [temp11], targeted= True)[0])
info = attack.wer_compute([target_transcription], [temp11], targeted= True)[1]
print(f"Insertion: {info[0][1]}, Substitution: {info[0][0]}, Deletion: {info[0][2]}")