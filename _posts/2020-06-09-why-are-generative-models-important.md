---
layout: post
title: Why are generative models important?
published: true
---


Within the Machine Learning community, generative models have become very popular over the last few years. These are generally based on Generative Adversarial Networks (Goodfellow et al. 2014), Variational Auto-encoders (Kingma & Welling, 2014), and more recently density models based on invertible flows (Dinh et al, 2016) or autoregressive models (Oord et al, 2016). These models all attempt the difficult task of estimating the distribution of high-dimensional objects such as images (milions of dimensions for each pixel and RGB channel), audio (each timestamp is a dimension), text (each letter is a dimension), genetic data (each nucleotide is a dimension). By estimating the distribution, we mean either the ability to generate samples (an image, a corpus of text) or to estimate the density of a given object (e.g. estimate p(image|model)).

While we won't go into the specific advantages of GANs, VAEs, etc ..., we will focus on the question: "Why do we need generative models?". For other ML problems such as classification or regression, which reduce a complex image to a label/number, the answer is obvious: we deploy them in a robot which makes decisions based on these predictions, or provide the answer to a person which makes a decision based on the prediction (e.g. doctor decides to give treatment X to a patient). But why would we like to go from low-dimensions (cat label) to high-dimensions (image of a specific cat)? We clarify some of these reasons below.
 

## Communication with users (Interpretability, Debugging, Fairness)

Generating images or audio are very useful for communicating with users. For example, the movie industry would be very interested to generate potential movie scenes from the text script, or maybe only parts of the movie such as the special effects. AI speech assistants ([OK Google](https://assistant.google.com/), [Alexa](https://developer.amazon.com/en-GB/alexa)) need to generate high-dimensional audio to communicate information back to the users. 

Communication with users is only useful for stimuli that we can easily percieve (images, sound, etc ...). For example, one could also generate high-dimensional fake genomic sequences, but this is less useful for humans since we cannot easily interpret or visualise them. 

### Interpretability


One can use generative models to understand what concept has been learned by an individual neuron -- see Fig. 1 ([Bau et al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf)). By activating that particular neuron or a set of neurons, the generator creates samples from that concept (Fig 2). The model can be assessed by the quality of the samples: 
* if generated samples are not realistic, the mapping between latent and image is likely wrong. The model might not have enough capacity, learning might not converge in reasonable time, etc ...
* if adding a tree in the image results in the sky changing color, then the model learned a spurious correlation. One can consider disentangled representations or augumenting the dataset with different combinations of trees and skies. 


![](https://i.imgur.com/NVW89oz.png)
*Fig 1. Images are used to show what class of concepts each neuron has learned. ([Bau et al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf))*

![](https://gandissect.csail.mit.edu/img/demo_short3.gif) 

*Fig 2. GAN-based generator used for painting and image manipulation -- see [gandissect.csail.mit.edu](https://gandissect.csail.mit.edu/) ([Bau et al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf))*


### Debugging

Generating images can help interpret what the model learned, and thus debug potential errors. After model training is complete, one can use a generative model to create hard-to-classify images, or corner-cases, and evaluate how the ML system works on these images. As exemplified in Fig 1., one can go a step further and cover a range of input images.  

![](https://i.imgur.com/fFBYtwn.jpg)
*Starting from a query image, one can generate artificial images that modify the classification score (from smiling=0.0 to smiling=1.0) without changing the identity of the subject. If the classifier learned a spurious correlation due to dataset bias (e.g. all smiling images had white background), the generator will also learn to modify the background color, which would be noticeable by the user. Image from ([Singla et al, 2020](https://openreview.net/pdf?id=H1xFWgrFPS))*


## Communication with other ML systems - The image as a "good interface"

Images, audio or other high-dimensional objects provide a good interface for machine learning systems to communicate with each other. This is especially true with incompatible systems that have different underlying representations. Imagine a transfer learning scenario where we'd like to transfer knowledge from a deep learning (DL) system to a kernel machine. 

Imagine that a DL system trained on Imagenet learned that all trees (input) are green (output). How can the DL system tell the kernel machine that all trees are green? Since the internal representations are all completely different, it is likely not possible to transfer and manipulate the weights from the DL system to the kernel machine. However, one can build a generator to generate many possible images of (input=trees,output=green), and then fine-tune the kernel machine based on the new data. If the target system has a good inductive bias, this form of data augmentation will enable learning of the new concept. In this case, the image is the *interface* between the two systems. 

It should be noted that other types of representations can constitute interfaces between ML or non-ML systems. For example, in neuroscience it is common for researchers to segment brain regions using e.g. a DL network, followed by 

## Simulation 




## Perform Bayesian Posterior Optimisation


## Anomaly detection



Next you can update your site name, avatar and other options using the _config.yml file in the root of your repository (shown below).

![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.
