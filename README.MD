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

## Example execution:
```
Enter your question: If it takes 3 towels 3 hours to dry in the sun, how long will it take 5 towels to dry?
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

Fused Answer from the three models:
*** Answer 1:
If it takes 3 towels 3 hours to dry in the sun, how long will it take 5 towels to dry?py
<|fim_prefix|><|fim_suffix|><|fim_middle|>#If it takes 3 towels 3 hours to dry in the sun, how long will it take 5 towels to dry?
# 3 * 3 = 9
# 5 * 3 = 15
# 15 / 9 = 1.66666666
*** Answer 2:
3 hours
*** Answer 3:
If it takes 3 towels 3 hours to dry in the sun, how long will it take 5 towels to dry? 

## Step 1: Calculate the drying time per towel
To find the drying time per towel, we need to divide the total drying time by the number of towels. In this case, it takes 3 hours for 3 towels.

## Step 2: Calculate the drying time per towel
Drying time per towel = total drying time / number of
```