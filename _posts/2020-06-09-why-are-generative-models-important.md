---
layout: post
title: Why are generative models important?
published: true
---


Within the Machine Learning community, generative models have become very popular over the last few years. These are generally based on Generative Adversarial Networks [(Goodfellow et al. 2014)](https://arxiv.org/abs/1406.2661), Variational Auto-encoders [(Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114), and more recently density models based on invertible flows [(Dinh et al, 2016)](https://arxiv.org/abs/1605.08803) or autoregressive models [(Oord et al, 2016)](https://arxiv.org/abs/1601.06759). These models all attempt the difficult task of estimating the distribution of high-dimensional objects such as images (milions of dimensions for each pixel and RGB channel), audio (each timestamp is a dimension), text (each letter is a dimension), genetic data (each nucleotide is a dimension). By estimating the distribution, we mean either the ability to generate samples (an image, a corpus of text) or to estimate the density of a given image or object -- i.e. $p(image \| model)$.

While we won't go into the specific advantages of GANs, VAEs, etc ..., we will focus on the question: "Why do we need generative models?". For other ML problems such as classification or regression, which reduce a complex image to a label/number, the answer is obvious: we deploy them in a robot which makes decisions based on these predictions, or provide the answer to a person which makes a decision based on the prediction (e.g. doctor decides to give treatment X to a patient). But why would we like to go from low-dimensions (cat label) to high-dimensions (image of a specific cat)? We clarify some of these reasons below.
 

## Communication with users

Generating images or audio are very useful for communicating with users. For example, the movie industry would be very interested to generate potential movie scenes from the text script, or maybe only parts of the movie such as the special effects. AI speech assistants ([OK Google](https://assistant.google.com/), [Alexa](https://developer.amazon.com/en-GB/alexa)) need to generate high-dimensional audio to communicate information back to the users. 

Communication with users is only useful for stimuli that we can easily percieve (images, sound, etc ...). For example, one could also generate high-dimensional fake genomic sequences, but this is less useful for humans since we cannot easily interpret or visualise them. 

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vTv4LW7o9d8fQHydsc6ieF3kkKrwUZud1FSqbLKIqTJHZiGxrFbj_joIz_nLcnMjnVVtofVl1r1VxYS/pub?w=576&amp;h=362">
</p>
*Fig X. Instead of the ML system performing both the prediction of disease progression and recommending the treatment, the ML system could only perform the future prediction, while the doctor can recommend the treatment.*
<br><br><br><br>



<p align="center">


<video width="480" controls>
  <source type="video/mp4" src="https://video.wixstatic.com/video/3d93c8_51f67029764f464b8a8bd0be615b7641/480p/mp4/file.mp4">
</video>
</p>
*Fig X. Demonstration of a model that predicts the evolution of Alzheimer's disease. From [daniravi.wixsite.com](https://daniravi.wixsite.com/researchblog/post/simulating-disease-progression-in-4d-mri) and [Ravi et al, 2019](https://arxiv.org/pdf/1907.02787.pdf)*


### Interpretability


One can use generative models to understand what concept has been learned by an individual neuron -- see Fig. 1 ([Bau et al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf)). By activating that particular neuron or a set of neurons, the generator creates samples from that concept (Fig 2). The model can be assessed by the quality of the samples: 
* if generated samples are not realistic, the mapping between latent and image is likely wrong. The model might not have enough capacity, learning might not converge in reasonable time, etc ...
* if adding a tree in the image results in the sky changing color, then the model learned a spurious correlation. One can consider disentangled representations or augumenting the dataset with different combinations of trees and skies. 


![](https://i.imgur.com/NVW89oz.png)
*Fig 1. Images are used to show what class of concepts each neuron has learned. ([Bau et al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf))*



<p align="center">
<img src="https://gandissect.csail.mit.edu/img/demo_short3.gif" > 
</p>
*Fig 2. GAN-based generator used for painting and image manipulation -- see [gandissect.csail.mit.edu](https://gandissect.csail.mit.edu/) ([Bau et al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf))*


### Debugging

Generating images can help interpret what the model learned, and thus debug potential errors. After model training is complete, one can use a generative model to create hard-to-classify images, or corner-cases, and evaluate how the ML system works on these images. As exemplified in Fig 1., one can go a step further and cover a range of input images.  

![](https://i.imgur.com/fFBYtwn.jpg)
*Starting from a query image, one can generate artificial images that modify the classification score (from smiling=0.0 to smiling=1.0) without changing the identity of the subject. If the classifier learned a spurious correlation due to dataset bias (e.g. all smiling images had white background), the generator will also learn to modify the background color, which would be noticeable by the user. Image from ([Singla et al, 2020](https://openreview.net/pdf?id=H1xFWgrFPS))*


## Communication with other ML systems - The image is the "interface"

Images, audio or other high-dimensional objects provide a good interface for machine learning systems to communicate with each other. This is especially true with incompatible systems that have different underlying representations. Imagine a transfer learning scenario where we'd like to transfer knowledge from a deep learning (DL) system to a kernel machine. 

Imagine that a DL system trained on Imagenet learned that all trees (input) are green (output). How can the DL system tell the kernel machine that all trees are green? Since the internal representations are all completely different, it is likely not possible to transfer the learned weights from the DL system to the kernel machine. However, one can build a generator to generate many possible images of (input=trees,output=green), and then fine-tune the kernel machine based on the new data. If the target system has a good inductive bias, this form of data augmentation will enable learning of the new concept. In this case, the image is the *interface* between the two systems. 

A special case of this is data augmentation. This can be done in order to induce invariances not already present in the model (e.g. rotations, scaling), or to augment corner-cases, such as accidents in self-driving car datasets (Fig. X). 

![](https://i.imgur.com/OBMh9pk.jpg)
*Fig X. Self-driving car datasets suffer from a long-tail of corner-cases: accidents are extremely rare. Generative models can be used to augment the dataset with rarely-seen events, such as accidents in rainy weather. Image from Carla simulator, [Dosovitskiy et al. 2017](http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)*

{% comment %}
### Any representation can be an interface

It should be noted that other types of representations can constitute interfaces between ML or non-ML systems. For example, in neuroscience it is becoming common for researchers to segment brain regions, lesions or tumours using e.g. a DL network, followed by another module that extracts features such as volume, thickness, etc ... which are used to compare across different populations. While one could learn an end-to-end mapping from brain image directly to the extracted features, the downside of this approach is that this mapping has to be re-learned every time a new feature of interest comes up (e.g. surface area, texture, etc ...). Splitting the mapping into different modules enables compositionality. 
{% endcomment %}


## Simulation 

Generative models can also be used as a world simulator or to generate parts of it. For example, if one can use generative models to generate "fake" humans in a 3D environment that could interact with a robot. Moreover, for each human, one could specify the desired age, gender and other desired demographic attributes. If a robot operates in the simulated world, the generative models can even be used to generate the action responses of the humans after the robot takes certain actions -- i.e. the updated world.  

<p align="center">
<img src="https://media0.giphy.com/media/l2SqgdbxYk5El9Rg4/source.gif" >
</p>
*Fig X. Conditional generative models can be used as a simulator of the world. The model generates images conditioned on the actions of the driver (left-right turns, pedals, etc ...). Inspired by [Jahanian et al, 2019](https://arxiv.org/pdf/1907.07171.pdf)*

## Perform Bayesian Posterior Optimisation

Some generative models based on invertible flow [(Dinh et al, 2016)](https://arxiv.org/abs/1605.08803) or autoregressive models [(Oord et al, 2016)](https://arxiv.org/abs/1601.06759) are able not only sample images, but to also estimate densities: given an input image, they can estimate the probability density function $f_{\theta}(image)$ in order to tell how likely it is that this image is under model family $f$ with parameters $\theta$. 

This allows us to use the generative model as a prior over the space of possible/realistic images. For example, imagine we're interested to perform an image colorization task by estimating *p(color\_img \| gray\_img)*, the distribution of all possible coloured images from a given grayscale image. We apply Baye's rule as follows:

$$p(color\_img|gray\_img) = p(gray\_img|color\_img) p(color\_img)$$

The first term, called the likelihood term, expresses the forward model and is very easy to compute: we take the colour image, convert it to grayscale, then compute a distance (L2) between the output and our input *gray\_img*. The second term is the prior term, and is most often assumed to be uniform. However, a uniform prior term would allow unrealistic combinations of colours to be given to the image. For example, a particular bird species could only appear with certain combinations of colors. This can be fixed by using a generative density model to constrain the inputs to only realistic images, in this case with realistic combinations of colors.   

<p align="center">
<img src="https://i.imgur.com/y2YDh7H.jpg=200x" width="70%"> 
</p>

*The same grayscale image can have multiple possible colorizations. The generative model can be used to estimate the prior term, which constraints the model to only plausible, realistic inputs. Image source: [Ardizonne et al, 2019](https://arxiv.org/pdf/1907.02392.pdf)*


### Anomaly detection

Here, we would like to detect if a new sample is out-of-distribution. For example, we have a dataset of healthy brain images, and we would like to detect if a new image is also healthy or not. Since there are a variety of brain pathologies we might not know a-priori, we cannot use a discriminative model which would classify between healthy vs disease X or disease Y. Therefore, we build a density model and then we can set a threshold on the density estimate -- if the new sample has density below the threshold, it is considered abnormal and flagged for a doctor to check it.


## Sharing the generative model instead of sharing the dataset

There are certain instances where the dataset cannot be shared, but a generative model would be able to be shared, or a dataset of fake samples generated with the model. This can occur due to privacy issues in medical data or copyright (with many caveats here). Dataset size can also be a consideration -- if the size of the generative model is smaller than the size of the dataset, one might prefer to transfer the smaller generative model over a network -- in this case, a form of compression has been achieved. 

