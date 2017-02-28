import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrate import undisort, loadCalMat
from line import Line
from thresholdBinary import *

from moviepy.editor import *

def pipeline(img, diskPkl, lines):
    #Calibrate camera & undistort img
    ## after calibratoin by running python calibrate.py
    #Extract mtx, dist from diskPkl
    mtx = diskPkl['mtx']
    dist = diskPkl['dist']
    undisImg = undisort(img, mtx, dist)

    #Color & Gradient Threshold 
    gradient_binaryImg = gradient_combined(undisImg,ksize=9)
    color_binaryImg = color_combined(undisImg)
    combin_binaryImg = np.zeros_like(gradient_binaryImg)
    combin_binaryImg[(color_binaryImg == 1)|(gradient_binaryImg == 1)] = 1

    #warpPerspective
    srcP, dstP = warpPtsCal(combin_binaryImg)
    warpedImg, Minv = warpImg(combin_binaryImg, srcP, dstP, (combin_binaryImg.shape[1], combin_binaryImg.shape[0]))

    #lane detection
    #extract lines
    llane = lines[0]
    rlane = lines[1]

    #If one of the lane is not detected, then do sliding window search
    if llane.detected == False or rlane.detected == False:
        windowImg, ploty = sliding_window_search(warpedImg, (llane,rlane))
    else:
        ploty = window_tracking(warpedImg, (llane,rlane))

    results = drawLineOnImg(warpedImg, ploty, (llane,rlane), undisImg, Minv)

    lcurRad, rcurRad, offset= curve_cal(ploty,(llane,rlane))


    cv2.putText(results, 'Left Radius: {:.0f}m'.format(lcurRad), (20,170),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA, False)
    cv2.putText(results, 'Right Radius: {:.0f}m'.format(rcurRad), (20,130),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA, False)
    cv2.putText(results, 'Offest to left: {:.0f}m'.format(offset), (20,100),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA, False)

    #plt.imshow(results)
    #plt.plot(llane.recent_xfitted[0], ploty, color='yellow')
    #plt.plot(rlane.recent_xfitted[0], ploty, color='yellow')
    #plt.show()
    return results



#Calculate srcPts and dstPts of the perspective
def warpPtsCal(img):
    y = img.shape[0]
    x = img.shape[1]
    srcP = np.float32([[x*0.43, y*0.67],     # top left
                     [x*0.58, y*0.67],      # top right
                     [x*0.85, y*0.98],      # bottom right
                     [x*0.17, y*0.98]])     # bottom left
    dstP = np.float32([[200, 0],        # top left
                      [x-200, 0],      # top right
                      [x-200, y],      # bottom right
                      [200, y]])       # bottom left
    return srcP, dstP

#do warpPerspective of the img with srcPts, dstPts, and imgSize
def warpImg(undisImg, srcP, dstP, imgSize):
    Mat = cv2.getPerspectiveTransform(srcP, dstP)
    Minv = cv2.getPerspectiveTransform(dstP, srcP)
    warpedImg = cv2.warpPerspective(undisImg, Mat, imgSize)
    return warpedImg, Minv

#draw line and filling area between lane on the undisorted img
def drawLineOnImg(warpedImg, ploty, lines, undisImg, Minv):
    llane = lines[0]
    rlane = lines[1]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpedImg).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([llane.recent_xfitted[0], ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rlane.recent_xfitted[0], ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Draw lines onto color_warp
    color_warp[llane.ally, llane.allx] = [255, 0, 0]
    color_warp[rlane.ally, rlane.allx] = [0, 0, 255]

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warpedImg.shape[1], warpedImg.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undisImg, 1, newwarp, 0.4, 0)

    return result


def sliding_window_search(binary_warped, lines):
    #Take a histogram of the bottom half of the img
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 15
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 70
    # Set minimum number of pixels found to re-center window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Define list of windows movements
    lwin_list = []
    dlwin_list = []#to save the delta of left lane base pos
    rwin_list = []
    drwin_list = []#to save the delta of right lane base pos


    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), thickness=4)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), thickness=4)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                          & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                           & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Append the base position to win_list
        lwin_list.append(leftx_current)
        rwin_list.append(rightx_current)
        # Append the delta increment to dwin_list, these delta value are used to 
        # guess the current lane base
        if len(lwin_list) > 1:
            dlwin_list.append(lwin_list[-1] - lwin_list[-2])
            drwin_list.append(rwin_list[-1] - rwin_list[-2])

        # If found > minpix pixels, re-center the next window to the mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        # If found < minpix pixels, guess the next window according to the last 5 movement of base pos
        else:
            dlwin_avg = sum(dlwin_list[-5:]) / 5
            leftx_current = int(leftx_current + dlwin_avg)

        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            drwin_avg = sum(drwin_list[-5:]) / 5
            rightx_current = int(rightx_current + drwin_avg)

    # Concatenate the arrays of indices
    # left_lane_inds is the list of all indices of pixels in all the windows
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Extract left lane & right lane
    llane = lines[0]
    rlane = lines[1]
    # If points are detected in the left or right lanes, update self.allx and self.ally
    # Left lane
    if (leftx.size == 0) | (lefty.size == 0):
        llane.detected = False
    else:
        llane.detected = True
        llane.allx = leftx
        llane.ally = lefty
    # Right lane
    if (rightx.size == 0) | (righty.size == 0):
        rlane.detected = False
    else:
        rlane.detected = True
        rlane.allx = rightx
        rlane.ally = righty

    # Fit a second order polynomial to each
    left_fit = np.polyfit(llane.ally, llane.allx, 2)
    right_fit = np.polyfit(rlane.ally, rlane.allx, 2)

    # Update the current coefficients
    llane.current_fit = left_fit
    rlane.current_fit = right_fit

    # Update the coefficient list
    llane.best_fit.append(left_fit)
    rlane.best_fit.append(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = llane.current_fit[0]*ploty**2 + llane.current_fit[1]*ploty + llane.current_fit[2]
    right_fitx = rlane.current_fit[0]*ploty**2 + rlane.current_fit[1]*ploty + rlane.current_fit[2]
    # Store the fitx values in the class for averaging purposes and plot
    llane.recent_xfitted.insert(0, left_fitx)
    rlane.recent_xfitted.insert(0, right_fitx)

    return out_img, ploty

def window_tracking(binary_warped, lines):
    """
    the same as sliding window search
    """
    llane = lines[0]
    rlane = lines[1]
    
    # Set the width of the curve +/- margin
    margin = 70

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Caluculate the curve values
    left_curve_vals = llane.current_fit[0]*(nonzeroy**2) + llane.current_fit[1]*nonzeroy + llane.current_fit[2]
    right_curve_vals = rlane.current_fit[0]*(nonzeroy**2) + rlane.current_fit[1]*nonzeroy + rlane.current_fit[2]

    # left_lane_inds is the list of all indices between the curve +- margin
    left_lane_inds = ((nonzerox > (left_curve_vals - margin))
                      & (nonzerox < (left_curve_vals + margin))) 
    right_lane_inds = ((nonzerox > (right_curve_vals - margin))
                       & (nonzerox < (right_curve_vals + margin)))  

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If points are detected in the left or right lanes, update self.allx and self.ally
    # Left lane
    if (leftx.size == 0) | (lefty.size == 0):
        llane.detected = False
    else:
        llane.detected = True
        llane.allx = leftx
        llane.ally = lefty
    # Right lane
    if (rightx.size == 0) | (righty.size == 0):
        rlane.detected = False
    else:
        rlane.detected = True
        rlane.allx = rightx
        rlane.ally = righty
    """
    above the same as sliding window search
    """

    # If the number of detected pixels is below a threshold,
    # run sliding windows on the next image.
    threshodPix = 5000
    if len(llane.allx) < threshodPix:
        llane.detected = False
    if len(rlane.allx) < threshodPix:
        rlane.detected = False

    # Fit a second order polynomial to each
    left_fit = np.polyfit(llane.ally, llane.allx, 2)
    right_fit = np.polyfit(rlane.ally, rlane.allx, 2)

    # Update the current coefficients with the weight of 0.7
    w = 0.7
    if len(llane.best_fit) > 1:
        llane.current_fit = (w*left_fit) + ((1-w)*(sum(llane.best_fit)/len(llane.best_fit)))
        rlane.current_fit = (w*right_fit) + ((1-w)*(sum(rlane.best_fit)/len(rlane.best_fit)))
    else:
        llane.current_fit = left_fit
        rlane.current_fit = right_fit

    # Update the coefficient list
    llane.best_fit.append(left_fit)
    rlane.best_fit.append(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = llane.current_fit[0]*ploty**2 + llane.current_fit[1]*ploty + llane.current_fit[2]
    right_fitx = rlane.current_fit[0]*ploty**2 + rlane.current_fit[1]*ploty + rlane.current_fit[2]
    # Store the fitx values in the class for averaging purposes and plot
    llane.recent_xfitted.insert(0, left_fitx)
    rlane.recent_xfitted.insert(0, right_fitx)

    return ploty

#calculate the curve rad and offset
def curve_cal(ploty, lines):
    #Extract lanes
    llane = lines[0]
    rlane = lines[1]
    # Define conversions in x and y from pixels space to meters
    y_eval = np.max(ploty)
    baseDiff = rlane.recent_xfitted[0][y_eval] - llane.recent_xfitted[0][y_eval]
    #lane length 30 m
    #lane width 3.7 m
    ym_per_pix = 30/y_eval # meters per pixel in y dimension
    xm_per_pix = 3.7/baseDiff # meters per pixel in x dimension
    

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, llane.recent_xfitted[0]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rlane.recent_xfitted[0]*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    llane.radius_of_curvature = left_curverad
    rlane.radius_of_curvature = right_curverad

    #Offset Cal
    carPos = y_eval/2
    lane_center = llane.recent_xfitted[0][y_eval] + (baseDiff/2)
    offset = (carPos - lane_center)*xm_per_pix
    llane.line_base_pos = offset

    #print(left_curverad, 'm', right_curverad, 'm', offset, 'm')
    return left_curverad, right_curverad, offset

if __name__ == "__main__":
    #load camera calibration matrix
    diskPkl = loadCalMat()
    llane = Line()
    rlane = Line()
    lines = (llane, rlane)
    
    #video output
    #fourcc = cv2.VideoWriter_fourcc(*'X264')
    #out = cv2.VideoWriter('project_video_output.mp4', -1, 20.0, (1280, 720))

    cap = cv2.VideoCapture('project_video.mp4')
    ret, frame = cap.read()
    count = 0
    while frame is not None:
        result = pipeline(frame, diskPkl,lines)
        ret, frame = cap.read()
        cv2.imwrite('./videos/'+str(count)+'.jpg', result)
        count += 1
        cv2.imshow('video',result)
        #out.write(result)
        #out.write(result)
        cv2.waitKey(2)
    cap.release()
    #cv2.destroyAllWindows()
    print ('done')