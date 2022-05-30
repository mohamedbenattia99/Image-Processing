import numpy as np
import random

def arrayToMatrix(Array, height, width,colour):
    if not colour:
        Matrix = []
        for h in range(height):
            Row = []
            for w in range(width):
                Row.append(Array[h * width + w])
            Matrix.append(Row)
        return Matrix
    else:
        return Array


def clone(Matrix):
    return [row[:] for row in Matrix]


def noise(Matrix, width, height, val):
    new_Matrix = clone(Matrix)
    for h in range(height):
        for w in range(width):
            x = random.randint(0, 20)
            if (x == 0):
                new_Matrix[h][w] = 0
            if (x == 20):
                new_Matrix[h][w] = val
    return new_Matrix


def ascii(image, width, height):
    new_image = clone(image)
    chars = ["B", "S", "#", "&", "@", "$", "%", "*", "!", ":", "."]
    for h in range(height):
        for w in range(width):
            new_image[h][w] = chars[new_image[h][w] // 25]
    return new_image
def matrixToArray(Matrix, height, width):
    Array = []
    for h in range(height):
        for w in range(width):
            Array.append(Matrix[h][w])
    return 

def generate_from_flat(width,height,flat_array):
    image =np.zeros((height,width),np.int32)
    for i in range(height):
        image[i] = flat_array[i*width:width*(i+1)]
    return image

def extract_from_struct(data):
    return data[0],data[1],data[2],data[3]