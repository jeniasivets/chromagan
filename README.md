## ChromaGAN Project

This project is based on the paper [ChromaGAN: A Conditional Generative Adversarial Network for Colorization](https://arxiv.org/pdf/1907.09837.pdf).

The project implements a model for the colorization of black and white images.

### Objective

The goal is to restore the two chromatic channels (a and b) of an image in the CIE Lab color space based on the L channel (luminance) of the image.

#### Generated Samples

Colorizing a photograph without ground truth data
<p float="left">
  <img src="images/0.png" width=45%>
  <img src="images/1.png" width=45%>
</p>
Colorizing test images from TinyImageNet
<img src="images/grid.png">

### Code

Training and inference scripts are available in the notebook as well as helper functions in `./scripts` directory.

### Network Architecture

The overall architecture consists of two main components: the generator and the discriminator. The generator takes as input a concatenated 3-channel L channel image and produces two outputs: the first output (a, b) represents the chromatic channels of the image, while the second output provides the class distribution of the input images according to ImageNet. The discriminator takes an image of size HxWx3 in the CIE Lab color space and returns the validity of the generated colorization.

**Detailed Architecture Overview**

Let’s take a look at the architecture diagram:

![Model Architecture](images/model.png)

The generator takes a 3xL black and white image as input. The initial layers of the generator replicate the VGG-16 architecture (depicted as yellow blocks), which consists of a sequence of convolutions followed by ReLU activations. After the yellow blocks, the outputs of the generator split into two streams. The first (purple) stream follows a Conv-BatchNorm-ReLU pattern, while the lower red stream contains Conv-BatchNorm-ReLU blocks that are divided: the gray section returns the class distribution for ImageNet, while the red section, composed of a sequence of linear layers (additional features from VGG-16), is reshaped to match the size of the purple block and concatenated with the main flow of the generator. This is followed by upsampling blocks, resulting in two color channels (a, b) at the generator's output.

The green part of the architecture diagram represents the discriminator, which is based on the PatchGAN architecture using 4x4 convolutions, BatchNorm, and LeakyReLU activations.

### Loss Functions

The training objective is the sum of three loss components: 
1. **Color Error Loss** – the L2 norm between the true chromatic channels and those generated.
2. **Adversarial Loss with GP** – the Wasserstein GAN loss with gradient penalty.
3. **Class Distribution Loss** – a classifier loss calculated as the KL divergence between the gray outputs of the generator and the predictions from a pre-trained classifier (using VGG-16).

### Model Training

Experiments were logged in Neptune.ai and can be viewed [here](https://ui.neptune.ai/calistro/chromagan/experiments?viewId=standard-view).

The authors of the paper trained the model on the full ImageNet dataset; however I would use a smaller `TinyImageNet` dataset for the training. To adapt the dataset to this architecture and task, I resized the images to 224x224 (the size the pre-trained classifier was trained on) and established a correspondence between the 200 classes in TinyImageNet and the 1000 corresponding classes in ImageNet. To obtain predictions from the pre-trained classifier on TinyImageNet, I used the predictions from ImageNet, after which I selected the 200 relevant logits from the resulting 1000-length logit vector, followed by applying Softmax.

To generate black and white images from the TinyImageNet dataset, a transformation from RGB to CIE Lab was performed to extract the (L, a, b) channels. This transformation was carried out using the `skimage` library and was applied to both the generator and the discriminator inputs (the discriminator takes the original image in CIE Lab format, while the generator’s output (a, b) is concatenated with the L channel of the input).

For the visualization of the generated colorizations during training, an inverse transformation from CIE Lab to RGB was also performed using `skimage`.

### Experiments

The conducted experiments included:

- training the generator and discriminator for 3 epochs with a learning rate of 2e-4 (PSNR was 23.5)
- training with a smaller learning rate of 2e-5 for 4 epochs (PSNR = 24.5)

PSNR was used as a quality metric.

The colorized images from the test set were logged each epoch. They can be viewed in the logs under the "colorization" column.

Additionally, an experiment was conducted on colorizing a black and white photograph without ground truth data, which includes a comparison with the service provided by Mail.ru at https://9may.mail.ru/restoration/.