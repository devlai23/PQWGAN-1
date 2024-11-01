# PQWGAN
This repository contains an implementation of PQWGAN, a hybrid quantum-classical GAN framework that uses quantum circuits to enhance image generation and manipulation tasks. PQWGAN leverages the distinct features of quantum mechanics, such as superposition and entanglement, within a generative adversarial network (GAN) to explore the role of quantum computing in GAN architectures.

# How to Run Through Command Line
The model takes in multiple hyperparameters like number of qubits and layers. An example to run the model is:
'python3 train.py --classes 01234 --dataset mnist --patches 14 --layers 15 --qubits 8 --batch_size 25 --out_folder results'

# Project Overview
PQWGAN integrates quantum circuits in both the generator and optionally the critic, allowing for enhanced data transformations and deeper exploration of GAN properties with quantum-enhanced layers. This hybrid model incorporates PennyLane for quantum circuit simulation and PyTorch for classical components, creating a versatile GAN structure that can be tuned across a range of quantum and classical parameters.

# Key Features
Hybrid GAN Architecture: PQWGAN's generator comprises multiple quantum circuits, each serving as a "sub-generator" responsible for a distinct patch of the output image. The critic can be either a classical or a quantum model, enabling flexible experimentation with different architectures.

# Quantum Generator:

Patch-Based Generation: The quantum generator divides image generation into patches, with each patch handled by a separate quantum circuit. This design enables parallelization of patch generation and adapts well to quantum circuit constraints on the number of qubits.
Quantum Circuit Configuration: Each quantum circuit is initialized with a specified number of data qubits, ancillary qubits, and layers of quantum gates.
Quantum Encoding Techniques:
Amplitude Encoding: Latent vectors are encoded using RY rotations, which map classical data onto quantum states. Each qubit receives a rotation based on values from the latent vector, introducing superposition states that represent combinations of features.
Angle Encoding: Variational rotations (RY, RZ) based on parameters (weights) add complexity to each qubit’s state, with each qubit's parameters optimized during training.
Entanglement with CNOT Gates: Each layer contains entangling CNOT gates between adjacent qubits to introduce quantum correlations, allowing the generator to capture non-linear relationships in the data. This entanglement pattern is applied across layers, propagating correlations that influence patch generation.
Variational Parameters: The trainable parameters (weights) control the rotation gates for each qubit in each layer, making the circuit adaptable to data distributions through gradient-based optimization.
Classical and Quantum Critics:

Classical Critic: The default critic, a neural network that evaluates real and generated images.
Quantum Critic (Optional): A quantum circuit-based critic can be specified, using a similar encoding and entanglement strategy as the quantum generator. This option allows testing PQWGAN’s full quantum capability in the adversarial framework.
Data Compatibility: PQWGAN initially trains on the MNIST and Fashion MNIST datasets, consisting of 28x28 grayscale images. These datasets are divided into patches to meet the constraints of limited qubits in quantum processing, offering a structured approach to integrate quantum components effectively.

Quantum Backend: PennyLane’s default.qubit quantum simulator is used to simulate quantum circuits, which is adaptable to a variety of backends. Future work could integrate hardware backends like IBM Q or Rigetti.

# Quantum Circuit Details
1. Quantum Generator Circuit (circuit function):
Input: Takes a latent vector and a set of trainable weights.
Encoding Latent Vector: Each element of the latent vector is encoded onto a qubit using RY rotations, initializing the qubits into a superposition representing a mix of features.
Parameterization and Entanglement:
Each layer consists of parameterized rotation gates applied to each qubit. The weights parameter controls the angles of these gates, creating tunable parameters that the model learns during training.
Entangling gates (CNOTs) are placed between adjacent qubits, which introduce quantum correlations, adding complexity to the circuit’s representational power.
Output: The circuit’s measurement is represented by the probability distribution of each state, computed via qml.probs, which provides a list of probabilities across qubit states.
2. Partial Trace and Post-Processing (partial_trace_and_postprocess):
Post-Measurement Processing:
Only the probabilities associated with a particular ancilla state are retained. This selective trace introduces a conditional measurement on the ancillary qubit, simulating a partial trace of quantum states, allowing for “collapsing” onto meaningful states.
Patch Normalization: The probability distribution is normalized and scaled to fall within a [-1, 1] range to produce pixel values compatible with GAN training. This ensures the generator output is in a format consistent with traditional GANs.
Getting Started

# Results:

Generated images, Wasserstein distance, and model checkpoints are saved to the output directory, with regular image samples saved for progress tracking.

# Training Details
Training is optimized with several techniques to improve GAN stability:

Gradient Penalty: A gradient penalty term is calculated for the critic to ensure stable training, which prevents the gradient from exploding or vanishing.
Wasserstein Distance Logging: Distance values are stored to assess training quality.
Epoch Timing: Each epoch duration is recorded to track training efficiency.
