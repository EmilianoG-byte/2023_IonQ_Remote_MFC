import qiskit
from qiskit import QuantumCircuit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import matplotlib.pyplot as plt
from numpy import linalg


images=np.load('../data/images.npy')
labels=np.load('../data/labels.npy')

dataset=list()
for i in range(50):
    dic={"image":images[i], "category":labels[i]}
    dataset.append(dic)

#define utility functions

def simulate(circuit: qiskit.QuantumCircuit) -> dict:
    """Simulate the circuit, give the state vector as the result."""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()
    
    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population
    
    return histogram


def histogram_to_category(histogram):
    """This function take a histogram representations of circuit execution results, and process into labels as described in 
    the problem description."""
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive


def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    counter = Counter([len(gate[1]) for gate in circuit.data])
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    #for i in range(2,20):
        #assert counter[i]==0
        
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(image1, image2)



n=len(dataset)
mse=0
gatecount=0

def half_image(image):
    length=int(len(image)/2)
    new_image=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            new_image[i,j]=np.sum(image[2*i:2*i+2,2*j:2*j+2])
    return(new_image)

def rescale_image(image):
    new_image=np.zeros((32,32))
    for i in range(28):
        for j in range(28):
            new_image[i+2,j+2]=image[i,j]
    return(half_image(new_image))

def vectorize(image):
    vector=image[0]
    for line in image[1:]:
        vector=np.concatenate((vector,line))
    vector=vector/(linalg.norm(vector))
    return(vector)

# Functions 'encode' and 'decode' are dummy.
def encode(image):
    new_image=rescale_image(image)
    print(new_image.shape)
    statevector=vectorize(new_image)
    circuit=QuantumCircuit(8)
    circuit.initialize(statevector)
    #circuit=circuit.decompose(reps=10)
    return(circuit)

def decode(histogram):
    image_data=np.zeros((16*16))
    for key in histogram.keys():
        image_data[key]=histogram[key]
    image=np.zeros((32,32))
    for i in range(16):
        for j in range(16):
            image[2*i,2*j]=image_data[i*16+j]
            image[2*i+1,2*j]=image_data[i*16+j]
            image[2*i,2*j+1]=image_data[i*16+j]
            image[2*i+1,2*j+1]=image_data[i*16+j]
    image=image[2:30,2:30]
    return image

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)
    #reconstruct the image
    image_re=decode(histogram)
    return circuit,image_re
