from IPython.display import Audio
from pydub import AudioSegment
import pylab
import sys
import torch
sys.path.append('./audio-diffusion')
from audiodiffusion import AudioDiffusion



device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)
model_id = "teticio/audio-diffusion-256" 
audio_diffusion = AudioDiffusion(model_id=model_id)
# mel is used for conversion 
mel = audio_diffusion.pipe.mel


image = Image.open("generated_spectro.jpg")
display(image)
sound = mel.image_to_audio(image)
display(Audio(sound, rate=22050))