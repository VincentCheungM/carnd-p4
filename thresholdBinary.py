import cv2
import numpy as np

"""
gradient threshold
"""
#Absolute of gradient
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, abs_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1

    # Return the result
    return binary_output

#Magnitude of gradient
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

#Direction of gradient
def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

    # Return the binary image
    return binary_output

#Gradient combined method
def gradient_combined(img, ksize=3, abs_thresh=(20, 100), mag_thresh=(30, 100), dir_thresh=(0.5, 1.3)):
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, abs_thresh=(20, 100))
    sybinary = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, abs_thresh=(20, 100))
    mag_binary = mag_threshold(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, dir_thresh=(0.5, 1.3))

    combined_gradient = np.zeros_like(sxbinary)
    combined_gradient[((sxbinary == 1) & (sybinary == 1)) | ((mag_binary == 1) & (dir_binary == 1))
         | ((sxbinary == 1) & (dir_binary == 1))] = 1
    return combined_gradient


"""
color threshold
"""
def R_threshold(img, threshold=(200,255)):
    R = img[:,:,2]#cv2.imread -> CV2.COLOR_BGR
    R_binary = np.zeros_like(R)
    R_binary[(R > threshold[0]) & (R <= threshold[1])] = 1
    
    return R_binary


def H_threshold(img, threshold=(15,100)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    H = hls[:,:,0]
    H_binary = np.zeros_like(H)
    H_binary[(H > threshold[0]) & (H <= threshold[1])] = 1

    return H_binary


def S_threshold(img, threshold=(90,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    S = hls[:,:,2]
    S_binary = np.zeros_like(S)
    S_binary[(S > threshold[0]) & (S <= threshold[1])] = 1

    return S_binary


#color combined method
def color_combined(img, R_thresh=(170, 255), H_thresh=(30, 100), S_thresh=(90, 255)):
    R_binary = R_threshold(img, R_thresh)
    H_binary = H_threshold(img, H_thresh)
    S_binary = S_threshold(img, S_thresh)

    combined_color = np.zeros_like(R_binary)
    combined_color[((S_binary == 1) & ((H_binary == 1) | (R_binary == 1) ))] = 1

    return combined_color
