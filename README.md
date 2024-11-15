# PQWGAN
This repository contains an implementation of PQWGAN, a hybrid quantum-classical GAN framework that uses quantum circuits to enhance image generation and manipulation tasks. PQWGAN leverages the distinct features of quantum mechanics, such as superposition and entanglement, within a generative adversarial network (GAN) to explore the role of quantum computing in GAN architectures.


# How to Run Through Command Line
The model takes in multiple hyperparameters like number of qubits and layers. An example to run the model is:
```
python3 train.py  --classes 013 --dataset mnist --patches 28 --layers 10 --qubits 8 --batch_size 25 --out_folder results
```
This command runs the model to generate 0,1,2,3,4 with 14 patches, 15 layers, 8 qubits, and a batch size of 25. It puts the generated images into a folder called results.


# Project Overview
PQWGAN integrates quantum circuits in both the generator and optionally the critic, allowing for enhanced data transformations and deeper exploration of GAN properties with quantum-enhanced layers. This hybrid model incorporates PennyLane for quantum circuit simulation and PyTorch for classical components, creating a versatile GAN structure that can be tuned across a range of quantum and classical parameters.


# Key Features
Hybrid GAN Architecture: PQWGAN's generator comprises multiple quantum circuits, each serving as a "sub-generator" responsible for a distinct patch of the output image. The critic can be either a classical or a quantum model, enabling flexible experimentation with different architectures.


# Quantum Generator Details  **will upload picture**
This circuit for the quantum generator involves applying gates depending on the number of qubits and layers specified through the command line parameters. For each qubit, a Ry rotation gate is applied, to transform classical bits into quantum. Then, depending on how many layers are specified, a Rotation gate with tunable parameters, followed by a CNOT gate entangling each qubit with one another is applied.


# Quantum Critic Details
TODO
