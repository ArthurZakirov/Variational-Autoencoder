# Autoencoder 
---

### Table of Contents
- [Description](#description)
- [How To Use](#how-to-use)

---

## Description

This project contains several code implementations of current research papers in the topic of Generative Models, specifically Autoencoders. The following chapter gives an overview over the different versions beeing implemented.

#### Autoencoder
<img src="https://sci2lab.github.io/ml_tutorial/images/autoen_architecture.png" width="600" height="300">

The standard autoencoder consists of an encoder that compresses the data into a latent representation and a decoder, that restores the data into it's original representation. The standard autoencoder is deterministic, meaning that it does not capture a latent probability distribution. In fact the only thing it does is dimensionality reduction.

#### Variational Autoencoder
<img src="https://lilianweng.github.io/lil-log/assets/images/vae-gaussian.png" width="600" height="300">
The Variational Autoencoders adds a probability distribution to the standard Autoencoder. Now it is not deterministic anymore. Mathematically speaking, the Encoder fulfills the role of the approximation of the posterior latent distribution conditioned on the data. The Decoder is mathematically speaking the data distribution conditioned on the latent variable.

- z-posterior true: intractible      <img src="https://render.githubusercontent.com/render/math?math=p(z|x)"> 

- z-posterior approximation: Encoder <img src="https://render.githubusercontent.com/render/math?math=q(z|x)">. 
 
- z-prior:                           <img src="https://render.githubusercontent.com/render/math?math=p(z|x)"> 
  
- x-posterior: Decoder               <img src="https://render.githubusercontent.com/render/math?math=p(x|z)">. 

Training the VAE requires to approximate the loss log likelihood of the full datat distribtuion since the Integral of the marginal distribtuion is intractible. Instead we maximize th ELBO:

<img src="https://render.githubusercontent.com/render/math?math=log(p(x)) \geq L = E[log(p(x|z))] - KL(q(z|x)-p(z))">

#### Conditional Variational Autoencoder
<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-981-15-1956-7_15/MediaObjects/492966_1_En_15_Fig1_HTML.png" width="500" height="400">
The Conditional Variational Autoencoder allows to condition the decoder not just on the latent variable, but additionally on an input of choice. This changes the loss function:

<img src="https://render.githubusercontent.com/render/math?math=log(p(x)) \geq L = E[log(p(x|z,c))] - KL(q(z|x)-p(z,c))">



---

## How To Use

#### Installation
You can clone this repository by running the following command.
```
git clone --recurse-submodules <repository cloning URL>
```
#### Environment Setup
First, we'll create a conda environment to hold the dependencies.
```
conda create --name AEpytorch python=3.8 -y
source activate AEpytorch
pip install -r requirements.txt
```
Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python -m ipykernel install --user --name AEpytorch --display-name "Python 3.8 (AEpytorch)"
```

#### Model Training
Model | Command
--- | --- 
AE | ```python train.py --model AE```
VAE | ```python train.py --model VAE```
CVAE | ```python train.py --model CVAE```

Besides the model type you can choose different types of decoders types using the parameter ```--decoder_type``` which gives 3 options:
Argument | help
--- | ---
```--decoder_type repeat_latent``` | Use latent variable as input for every timestep
```--decoder_type concat_output_latent``` | Use latent as input for every timestep concatenated with the hidden state of the previous timestep
```--decoder_type output_as_input``` | Use hidden state of the previous timestep



#### Tensorboard
The tensorboard can be called via the following command.
```tensorboard --logdir```
