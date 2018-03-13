# StyleNet

A cute multi-layer LSTM network that can perform like a human 🎶 It learns the dynamics of music! The architecture was specifically designed to handle music of different genres.

If you wish to learn more about my findings, then please read my [blog post](http://imanmalik.com/cs/2017/06/05/neural-style.html) and paper:

> **Iman Malik, Carl Henrik Ek, [*"Neural Translation of Musical Style"*](https://arxiv.org/abs/1708.03535), 2017.**

![GitHub Logo](http://imanmalik.com/assets/img/stylenet.png)



## Prerequisites
You will need a few things in order to get started. 

1. Tensorflow
2. mido
3. pretty_midi
4. fluidsynth

## The Piano Dataset
I created my own dataset for the model. If you wish to use the Piano Dataset 🎹 for academic purposes, you can download it from [here.](http://imanmalik.com/assets/dataset/TPD.zip) The Piano Dataset is distributed with a [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). If you use this dataset, please reference this [paper](https://arxiv.org/abs/1708.03535):

  

## How to Run
``` python main.py -current_run <name-of-session> -bi ```

Flags:  
`-load_last` : Loads and continues from last epoch.  
`-load_model`: Loads specified model.  
`-data_dir` : Directory of datasets.  
`-data_set` : Dataset name.  
`-runs_dir` : Directory of session files.  
`-forward_only` : For making predictions (not training).  
`-bi` : If you wish to use bi-directional LSTMs. (HIGHLY recommended)

## Files
`pianoify.ipynb` : This was used to ensure the files across the dataset were consistent in their musical properties.  
`generate_audio.ipynb` : This was used to make predicitions using StyleNet and generate the audio.  
`convert-format.rb` : This was used to convert format 1 MIDIs into format 0.  
`file_util.py` : This contains folder/file-handling functions.  
`midi_util.py` : This contains MIDI-handling functions.  
`model.py` : StyleNet's Class.  
`data_util.py` : For shuffling and batching data during training.  

