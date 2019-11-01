

# Voice-Conversion-CycleGAN

### Paper and Dataset

**Paper：**[Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1711.11293)

**Dataset：**[VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211)

<br/>

### Model Structure

![image](https://github.com/alpharol/Voice_Conversion_CycleGAN/raw/master/figure/CycleGAN.png)

*Note: The channel of residual block may be wrong.*

<br/>

### Dependencies

Ubuntu 1080ti

python 3.5

tensorflow 1.14.0

PyWorld 0.2.8

numpy 1.15.4

librosa 0.5.1

<br/>

### File Structure

```bash
|--convert.py
|--model.py
|--module.py
|--preprocess.py
|--train.py
|--utils.py
|--data--|vcc2016_training
       --|evaluation_all
```

<br/>

### Usage

#### Preprocess

```python
python preprocess.py
```

This process may take 13 minutes if using the same datasets.

<br/>

#### Train

```python
python train.py
```

It may take 5 minutes for one epoch. In order to get a good voice quality, 800 epoches are needed. 

If the other speakers are involved, please change the directory below.

```bash
train_A_dir_default = './data/vcc2016_training/SF1'
train_B_dir_default = './data/vcc2016_training/TM1'
```

<br/>

#### Inference

```python
python convert.py
```

The converted voice can be found in the directory below:

```bash
|--converted_voices
```

<br/>

### To-Do 

- [ ] Provide some converted samples.

