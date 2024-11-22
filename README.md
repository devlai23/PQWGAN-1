# Title

# Introduction

Motivation: 
A traditional Generative Adversarial Network (GAN) is a machine learning model composed of two neural networks: A Generator (creates fake data), and a Discriminator (distinguishes between real and fake data). The 2 Neural Networks compete in a feedback loop until the generator is able to create fake data that is seemingly indistinguishable from real data. 

Traditional GANs measure the difference between real and fake data using the Jensen-Shannon Divergence. However, this method of measurement becomes problematic when the distributions of fake and real data have similarities. This can lead to unstable training and limited diversity in generating outputs. 

From this Wasserterian GANs (WGANs) came into the picture of improving the way traditional GANs measure difference between real and fake data by using the Wasserstein Distance. The Wasserstein Distance is a metric that calculates the minimum cost of transforming the fake probability distribution into the real probability distribution by optimizing the cost function that considers the amount of probability mass moved in the distance it travels in the feature space. 

However, WGANs rely on deep neural networks to minimize the Wasserstein Distance. This poses challenges such as high computational costs, larger parameter spaces, and scalability issues for higher dimensional datasets. 

Quantum Advantage:
Quantum Machine Learning offers a fundamentally new computational paradigm that addresses the limitations of WGANs in terms of computational cost, scalability, and representation efficiency by using a Quantum Wasserstein Adversarial Network (QWGAN).

Unlike a traditional GAN, In a QWGAN, the generator which creates fake data is replaced by a Quantum Computer. This quantum generator uses special properties of quantum systems such as superposition (representing multiple probabilities at once) and entanglement (connections between quantum states), to generate data more efficiently and with greater diversity. 

QWGANs leverage quantum properties like superposition and entanglement to allow QWGANs to explore complex and diverse data distributions more efficiently by leveraging the exponentially larger Hilbert Space of quantum systems. Superposition enables the quantum generator to represent multiple probabilities or configurations seemingly simultaneously, while entanglement creates the intricate correlations  between qubits that capture the complex patterns within the data. These capabilities make QWGANs particularly powerful for generating data with intricate relationships and higher-dimensional features. 

However this efficiency can also become a drawback when QWGANs are trained on idealized datasets, which are often clean, consistent, and lack real-world variability. Because the quantum generator is highly efficient at modeling the exact distribution it is trained on, it may overfit to the training data, learning to replicate specific patterns/relationships without understanding the underlying structure of the data. This overfitting can result in poor performance when QWGANs are exposed to noisy, distorted, or incomplete data that deviates from the idealized training set.

To address this limitation, we incorporated data augmentation techniques to simulate real-world imperfections, such as noise and distortions, which challenge the model to learn more robust and invariant features.

Objectives:
Build a Quantum Wasserstein GAN (QWGAN) by integrating a Parameterized Quantum Circuit (PQC) as the generator and a classical neural network as the critic, combining the strengths of quantum computing and classical optimization.
Apply the QWGAN framework to simple datasets, such as MNIST (handwritten digits) to evaluate its ability to generate realistic and diverse outputs as a proof of concept.
Incorporate Data Modification Techniques such as Elastic Transforms, Gaussian noise, and Poisson noise to test the robustness of the QWGAN framework. 
Compare QWGAN performance with such data modification techniques against one another. 

# How to Run Through Command Line
The model takes in multiple hyperparameters like number of qubits and layers. An example to run the model is:
```
python3 train.py  --classes 013 --dataset mnist --patches 14 --layers 17 --qubits 6 --batch_size 25 --out_folder results
```
This command runs the model to generate 0,1,3, with 14 patches, 17 layers, 6 qubits, and a batch size of 25. It puts the generated images into a folder called results.

# Methods
The quantum framework that we used in this project is Pennylane. 

The architecture of the model is as follows. First, random vectors are sampled from a latent space, to be fed into the quantum generator. The quantum generator is the main component for generating the images. The quantum generator is split into many sub-generators that create “patches” of the image, a parameter that can be tweaked for performance. Each sub-generator contains a quantum circuit that processes part of the image. After going through the quantum circuit, the patches are then combined together to form a full picture. The generated sample is then compared to a “real sample,” an image taken directly from the dataset. Using a critic (neural network) the generated and real samples are compared for accuracy. The unit for accuracy is the Wasserstein Distance, helping us quantify how similar the produced images are to actual ones. The generator then receives feedback from the critic every cycle, tweaking its parameters to lower the Wasserstein Distance. The critic also simultaneously trains itself to better assess differences between real and generated samples.

The following is the quantum circuit used in each quantum sub-generator.

The circuit must first translate the latent vector mentioned earlier into a superposition state, which is done by applying a quantum Ry gate (Y Rotation). After this initialization the qubit will then undergo a series of parametrized rotation gates, determined by the # of layers specified by the user. This process is repeated a number of times. For each iteration, a rotation is applied using a number of parameters that will be tweaked in collaboration with the generator throughout the entire training process. After each patch has a rotation gate applied, they are then entangled with one another using a CNOT gate. This creates correlations between each patch. After this process, each patch is combined back together to form the final image.

# Dataset and Preprocessing
Data Description: Provide key details about the dataset (source, size, features, etc.).
The dataset used for this project was MNSIT, a widely used benchmark dataset for image recognition tasks. 
Source: National Institute of Standards and Technology (NIST).
Size: 70,000 grayscale images, split into training set of 60,000 images, and a testing set of 10,000 images 
Features: Each image is 28x28 pixels, representing handwritten digits spanning 0-9.
Pixel Values: Grayscale intensity values range from 0 (black) to 255 (white).  
 
Preprocessing Steps: Outline any data preprocessing techniques applied (normalization, feature scaling, dimensionality reduction).
To prepare the dataset for training the Quantum Wasserstein GAN (QWGAN), the following preprocessing steps were applied:
Normalization:
Pixel values were scaled to the range of [0,1] by dividing by 255. This was done to ensure the uniformity of input data and stabilizes the training process
Incorporating Data Modification Techniques:
To evaluate the robustness and adaptability of the QWGAN, the dataset was augmented with the following techniques:

For Elastic Transforms:
Definition: Elastic Transforms apply spatial distributions to an image by remapping the pixel locations based on smoothed displacement fields. These transforms are intended to mimic real-world deformations, such as stretching, twisting, or wrapping, while preserving the overall structure of the image. 
Purpose: To simulate natural variations in image shapes and test the model's ability to generalize beyond perfectly aligned data, and encourage the generator to learn invariant features that remain robust under shape changes.
Implementation:
Random values (dx and dy) are created for each pixel using a random state, representing the displacement along the x and y-axes. These values are scaled by α to control the intensity of the distortion.
The displacements are passed through a Gaussian filter with a standard deviation (𝜎) to ensure the distortions are smooth and realistic, avoiding abrupt pixel shifts.
A grid of original pixel indices (x and 𝑦) is created to serve as the base for displacement calculations.
The new pixel locations are calculated by adding the smoothed displacements to the original coordinates. The map_coordinates function remaps the pixel values to these new positions, creating the warped effect.

For Gaussian Noise:
Definition: Adds noise with a probability density function equal to a normal distribution
Purpose: The goal was to add light gaussian noise to the input during training in order to have more varied samples for a wider variety of outputs. The hope was that it would help the model output better looking numbers sooner.
Implementation: For every black and white pixel in the image, noise is sampled from a normal distribution and added to each pixel value

For Poisson Noise: Definition: Poisson noise is a type of noise that arises from the random nature of events. In real world scenarios this could include photons hitting a sensor in a variable manner
Purpose: Simulating poisson noise is commonly used to mimic the noise you might get from real-world data and imaging systems. 
Implementation: For every pixel (I) in the image, noise is sampled from a Poisson distribution (N). Then the pixel value is replaced with I + N. The Poisson distribution essentially applied slight, random variations to the number around its original value. 

# Results
The results of running our model are promising. There are several things to note. One, the Wasserstein distance starts low and increases. This is the phase in which the critic is still learning to differentiate between real and generated samples during the early stages. The generator also initially produces poor quality results with lots of noise, which the critic distinguishes as a high Wasserstein distance. As the model continues running, the generator improves but with fluctuations that are common in GAN training. This specific example was run with the following command line parameters: 28 patches, 5 qubits, 10 layers. Possible implications of this setup include a high number of patches being reflected in the stability of the training process, since the image generation is split into smaller, more manageable sections. 5 qubits on the other hand is on the lower side, and increasing the count could allow the generator to learn more complex features.

Adding noise: Results TBA

# Conclusion
Summary: 
Classical GANs are producing extremely realistic results with QGANs still far away from those levels. However, QGANs are still a promising area of research. This paper shows a method of generating 28 x 28 pixel images of handwritten numbers. We successfully generated a three class subset of the MNIST dataset: 0/1/3. Even though classical GANs have been able to do this, we achieved this task with three orders of magnitude fewer trainable parameters in the generator. Furthermore, we found that by adding noise to a percentage of the training samples ……

Impact: These findings further the promise of quantum machine learning by showing that it has huge potential against classical models with billions of parameters. If a quantum version could be created, it has the chance to decrease the resources needed by multiple magnitudes. Quantum machine learning could be the solution to problems unsolvable before due to a restriction of technological capabilities.

Future Work: Suggest potential improvements or extensions to your research.
We suggest a future direction in applying quantum generators to more complex and higher-resolution images. We were restricted in our computing power and were only capable of training models on 28 x 28 black and white pixel images.

# References
Tsang, Shu Lok, et al. “Hybrid Quantum-Classical Generative Adversarial Network for High Resolution Image Generation.” IEEE Transactions on Quantum Engineering, vol. 4, 2023, pp. 1–19, https://doi.org/10.1109/tqe.2023.3319319.