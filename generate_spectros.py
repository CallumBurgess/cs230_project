import numpy as np 
import os
from IPython.display import Audio
from pydub import AudioSegment
import pylab
import torch
from glob import glob 
import sys 
sys.path.append('./audio-diffusion')
from audiodiffusion import AudioDiffusion

genre_path = './Dataset/Data/genres_original'
data_path = './Dataset/Data'
file_path = [os.path.join(genre_path,x) for x in os.listdir(genre_path)]
genres = [x for x in os.listdir(genre_path)]

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)
model_id = "teticio/audio-diffusion-256" 
audio_diffusion = AudioDiffusion(model_id=model_id)
# mel is used for conversion 
mel = audio_diffusion.pipe.mel


sample_rate = 22050
audio_files = glob(genre_path + "/*/*.wav")
spec_to_label = dict()
for g in genres: 
  dirpath = "spectrograms/" + g
  os.mkdir(dirpath)

count = 0

#convert each image into 5s spectrograms 
for s in range(len(audio_files)):
  s = audio_files[s]
  if count % 100 == 0:
    print(count/1000)
  num = s[-9:-4]
  genre = s[len(genre_path) + 1: s.index("/", len(genre_path) +2)]
  try: 
    mel.load_audio(s)
  except:
    print("error loading: ", s)
    continue

  for i in range(mel.get_number_of_slices()):
    img = mel.audio_slice_to_image(i)
    img.save("./spectrograms/" + genre + "/" + num + str(i) + ".jpg" ) 
  count += 1