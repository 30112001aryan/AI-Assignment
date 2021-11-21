# Deep Learning project assignment Saarthi.ai

## Dependencies
PyTorch, torchaudio, numpy, soundfile, pandas, tqdm, textgrid.py

## Training
```
python main.py --train --config_path=<path to .cfg>
```

_ASR pre-training:
```
python main.py --pretrain --config_path=<path to .cfg>
```

## Inference

```python
import data
import models
import soundfile as sf
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg"); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device)) # load trained model

signal, _ = sf.read("test.wav")
signal = torch.tensor(signal, device=device).float().unsqueeze(0)

model.decode_intents(signal)
```
The ```test.wav``` file included with this repo has a recording of me saying "Hey computer, could you turn the lights on in the kitchen please?", and so the inferred intent should be ```{"activate", "lights", "kitchen"}```.

