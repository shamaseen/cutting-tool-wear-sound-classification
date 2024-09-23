import torch
import torchaudio

class Preprocessing():
    def __init__(self,n_fft:int=1024,win_length:int=None,hop_length:int=512,n_mels:int=64,target_sample_rate:int=16000,num_samples:int=160000):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_sample_rate=target_sample_rate
        self.num_samples=num_samples
        self._load_transform()

    def _load_transform(self):
        self.melspectogram=torchaudio.transforms.MelSpectrogram(sample_rate=self.target_sample_rate,n_fft=self.n_fft,hop_length=self.hop_length,n_mels=self.n_mels,win_length=self.win_length,normalized=True)
        self.AmplitudeToDB = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)

    def _mix_down(self,waveform):
        # convert to 1 channcel
        # used to handle channels
        waveform=torch.mean(waveform,dim=0,keepdim=True)
        return waveform
    
    def _resample(self,waveform,sample_rate):
        # used to handle sample rate
        if sample_rate != self.target_sample_rate:
            resampler=torchaudio.transforms.Resample(sample_rate,self.target_sample_rate)
            return resampler(waveform)
        return waveform
    
    def _cut(self,waveform):
        # cuts the waveform if it has more than certain samples
        if waveform.shape[1]>self.num_samples:
            waveform=waveform[:,:self.num_samples]
        return waveform
    
    def _right_pad(self,waveform):
        # pads the waveform if it has less than certain samples
        signal_length=waveform.shape[1]
        if signal_length<self.num_samples:
            num_padding=self.num_samples-signal_length
            last_dim_padding=(0,num_padding) # first arg for left second for right padding. Make a list of tuples for multi dim
            waveform=torch.nn.functional.pad(waveform,last_dim_padding)
        return waveform
    
    def _load_audio(self,path):
        for file in path:
            waveform, sample_rate = torchaudio.load(file, normalize=True)
            waveform=self._resample(waveform,sample_rate)   
            waveform=self._mix_down(waveform)
            waveform=self._cut(waveform)
            waveform=self._right_pad(waveform).squeeze(0) 
            waveform=torch.stack([waveform,waveform,waveform],dim=0)
            yield waveform
    def generate_sample(self,path):
        waveform=list(self._load_audio(path))
        waveform=torch.stack(waveform)
        # normalizeation
        waveform=self.melspectogram(waveform)
        # AmplitudeToDB
        waveform=self.AmplitudeToDB(waveform)
        # Noramalization
        waveform = (waveform - waveform.mean()) / waveform.std() 
        return waveform.to(torch.float32)



