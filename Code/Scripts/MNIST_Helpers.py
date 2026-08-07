import numpy as np
import matplotlib.pyplot as plt
import math

### MNIST Helpers
def Data_Binarizer(Dataset, Threshold=0):
    """Binarizer for the MNIST Dataset
    - The MNIST dataset is not quantized.
    - This converts the data into a binary -1,1 set 
    - Anything below the threshold is -1
    - Anything above is 1
    - Threshold of 0 is adequate. """
    Data_larger_than_threshold = 2*(Dataset>Threshold) -1 
    return Data_larger_than_threshold


def Vector_Array_to_Image_Array(Vector):
    """Converts an unfurled vector into an image array
    - vector of Length 784 -> (28 X 28) image"""
    if np.shape(np.shape(Vector)) == (1,) :
        ## Then this can be made into image
        Length = len(Vector)
        dim = int(math.sqrt(Length))
        V_image = Vector.reshape(dim,dim)

    elif np.shape(Vector)[0] == np.shape(Vector)[1]:
        ## this is already an image
        V_image = Vector
    return V_image

def Image_given_Vector(Vector, Label= None,  figure_size= 3,Verbose = False):
    """Plots the Image Given the Vector 
    - If label is provided, it is used for the title """
    Image_array = Vector_Array_to_Image_Array(Vector)
    plt.figure(figsize=(figure_size,figure_size))
    plt.imshow(Image_array)
    if Label != None:
        # we also have a label
        plt.title("This is a "+ str(Label))
    plt.show
    if Verbose==True:
        return Image_array
 

def quantize_images(images, n_bins):
    """
    Quantize images into n bins.
    
    Args:
        images: numpy array with values in [-1, 1]
        n_bins: number of bins
    
    Returns:
        Quantized images with values in {0, 1, ..., n_bins-1}
    """
    # Clip to ensure values are in [0, 1]
    
    Max_val = np.max(images, axis = 1)
    Min_val = np.min(images, axis = 1)
    Diff = Max_val-Min_val

    Images_in_0_1 = (images - np.reshape(Min_val, shape=(-1,1)))/ (np.reshape(Diff, shape=(-1,1)))
    
    # Scale to [0, n_bins] and floor
    quantized = np.floor(Images_in_0_1 * n_bins).astype(int)
    
    # Handle edge case where value = 1.0 exactly
    quantized = np.clip(quantized, 0, n_bins - 1)
    
    return quantized
