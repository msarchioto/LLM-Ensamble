# LLM Ensamble
In theory, using multiple LLMs you improve the performances by combining the answers

## Install deps
```
pip install -r requirements.txt
```

## Test if CUDA is available
```
(venv) user@userpc:~/LLM-Ensamble$ python3
Python 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```