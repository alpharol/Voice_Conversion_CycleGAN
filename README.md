

# Voice-Conversion-CycleGAN

### Paper and Dataset

**Paper：**[CycleGAN-VC: Non-parallel Voice Conversion Using Cycle-Consistent Adversarial Networks  ](https://ieeexplore.ieee.org/abstract/document/8553236?casa_token=nEkt-SBQoPoAAAAA:9VLqcVdeP_O4Cuhr6GLreLo8Y8Ph1eo0SGdVwd_24Dq0PaAEnEmjIpZQ6PkulcoH92zcbL4)

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

This process may take 13 minutes if using the same datasets. The processed data is stored in the directory `training_data`.

<br/>

#### Train

```python
python train.py
```

It may take 1 minutes for one epoch. In order to get a good voice quality, 500 epoch are needed. 

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



- [x] Separate the preprocess from the training steps.
- [x] Add process bar to the code.
- [x] Accelerate the training speed through saving the models by 100 epoch.
- [x] Add the training module based on the pretrain model.
- [x] Add the module of saving the last five epoch.
- [x] Provide some converted samples.

