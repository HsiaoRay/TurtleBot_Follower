
import numpy as np
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('output_dnn_nooverlay.avi')
# Print video resolution
w = int(cap.get(3))
h = int(cap.get(4))
print('Width: ' + str(w))
print('Height: ' + str(h))
fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS: ' + str(fps))

# Decrease FPS by N
N = 5

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Be sure to use the lower case
out = cv2.VideoWriter('output.avi', fourcc, int(fps/N), (2*w, h))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 4000,
                       qualityLevel = 0.1,
                       minDistance = 1,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

K = np.zeros((3,3))
K[0,0] = w/2
K[1,1] = h/2
K[2,2] = 1
K[0,2] = w/2
K[1,2] = h/2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Get initial features to track
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

# Initial R and t
#R_pos = np.eye(3)
R_pos = np.zeros((3,3))
R_pos[0,0] = 1
R_pos[2,2] = 1#-1
R_pos[1,1] = 1#-1
t_pos = np.zeros((3,1))
t_pos_hist = np.zeros((1800,3))
R_pos_prev = R_pos
t_pos_prev = t_pos

# Initial map
map_len = 125
wmap = 255*np.ones((2*map_len+1,2*map_len+1))
Nmap = 75
Mmap = 10

# Saved Frame number
t = 1
# Actual Frame Number
fn = 0

while(cap.isOpened()):
    # Decrease FPS by N
    for n in np.arange(N):
        ret, frame = cap.read()
        fn += 1

    

    if ret==True:
        # Our operations on the frame come here     
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
        
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        
        mask_use = np.zeros_like(gray)
        
        big_area = 0
        big_center = 320
        detected = 0
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
                object_type = detections[0,0,i,1]
                confidence = detections[0, 0, i, 2]
                if object_type == 15 and confidence > 0.2:
                    
                # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
        
                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format('person',confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),[0,0,255], 2)
                    cv2.rectangle(mask_use, (startX, startY), (endX, endY),1, -1)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
                    
                    rect_center = int((startX+endX)/2)
                    rect_area = (endX-startX)*(endY-startY)
                    detected = 1
                    if rect_area > big_area:
                        big_area = rect_area
                        big_center = rect_center

        v = 0
        if detected:
            if big_area > 25000:
                target_center = 320
                target_area = 150000
                kr = .002
                w_comm = -kr*(big_center-target_center)
                kt = 0.0000045
                v = -kt*(big_area - target_area)
                maxv = 0.25
                v = np.max([-maxv, v])
                v = np.min([maxv, v])
                #print(v)
                # Send Velocity command to turtlebot
                  
        
        # New features to track if less than 400
        if p0.shape[0] < 100:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
        
        # KLT Tracker using optical flow
        p0 = p0.astype('f')
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params) 
            
        # Select good points
        good_new_all = p1[st==1]
        good_old_all = p0[st==1]
        # Points shouldn't be detected on people since they can move
        st_mask = np.zeros(np.sum(st))
        for i in range(np.sum(st)):
            row_idx = good_new_all[i,1]
            row_idx = np.max([0, row_idx])
            row_idx = np.min([h-1, row_idx])
            row_idx = int(row_idx)
            col_idx = good_new_all[i,0]
            col_idx = np.max([0, col_idx])
            col_idx = np.min([w-1, col_idx])
            col_idx = int(col_idx)
            if mask_use[row_idx, col_idx] == 0:
                st_mask[i] = 1
        good_new = good_new_all[st_mask==1]
        good_old = good_old_all[st_mask==1]
        mask_use = 255*mask_use
            
        
        # Draw feature points
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            frame_draw = cv2.circle(frame,(int(a),int(b)),5,[0,0,0],-1)

            
        # Find Essential Matrix and recover pose
        E, mask = cv2.findEssentialMat(good_old, good_new, K, method=cv2.RANSAC, prob=0.999, threshold=0.1)
        tmp, R, T, tmp2 = cv2.recoverPose(E, good_old, good_new, K)
            
        T = T*v/(30/N)
        
        # Update total R and t
        R_pos = np.matmul(R,R_pos)
        t_pos = t_pos + np.matmul(R_pos.T,T)
        t_pos_hist[t,:] = np.squeeze(t_pos)
        
        wmap = wmap + Mmap
        
        proj1 = np.concatenate((np.eye(3),np.zeros((3,1))),axis=1)
        proj1 = np.matmul(K,proj1)
        proj2 = np.concatenate((R,T),axis=1)
        proj2 = np.matmul(K,proj2)
        pts4D = cv2.triangulatePoints(proj1, proj2, good_old.T, good_new.T)
        pts3D = pts4D[:3,:]/np.repeat(pts4D[3,:], 3).reshape(-1, 3).T
        pts3D_wf = np.matmul(R_pos.T,pts3D) - t_pos
        pts2D = pts3D_wf[[0,2],:]
        Npts = np.shape(pts2D)[1]
        for n in np.arange(Npts):
            xval = int(pts2D[0,n]/4)
            zval = int(pts2D[1,n]/4)
            if xval < map_len and xval > -map_len and zval < map_len and zval > -map_len:
                wmap[xval+map_len,zval+map_len] = wmap[xval+map_len,zval+map_len] - Nmap
                
        wmap = np.clip(wmap,0,255)
        wmap_resize = cv2.resize(wmap.astype(np.uint8), (w, h)) 
        wmap_show = np.zeros([h,w,3],dtype = np.uint8)
        wmap_show[:,:,0] = wmap_resize
        wmap_show[:,:,1] = wmap_resize
        wmap_show[:,:,2] = wmap_resize
        
        # Plot trajectory every 5 frames
        if np.mod(t,1) == 0:
            fig = plt.figure(3,figsize=(5, 4))
            fig.tight_layout(pad=0)
            plt.ion()
            plt.clf()
            N_avg = 8
            x_pos = np.convolve(t_pos_hist[:(t+1),0], np.ones((N_avg,))/N_avg, mode='valid')
            z_pos = np.convolve(t_pos_hist[:(t+1),2], np.ones((N_avg,))/N_avg, mode='valid')
            plt.plot(x_pos,z_pos)
            
            plt.xlim((-1.5,.5))
            plt.ylim((-1,1))
            plt.xlabel('x')
            plt.ylabel('z')
            plt.pause(0.001)
            
            plt_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            plt_image = plt_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # Resize to match frame
            plt_image = cv2.resize(plt_image,None,fx=1.28, fy=1.2, interpolation = cv2.INTER_AREA)

        
        # Increment frame counter
        t += 1
            
        # Update prev_frame and tracking points    
        prev_frame = frame
        prev_gray = gray
        p0 = good_new.reshape(-1,1,2)
        R_pos_prev = R_pos
        t_pos_prev = t_pos

        # write the frame
        frame_combine = np.zeros([h,2*w,3],dtype = np.uint8)
        frame_combine[:,:w,:] = frame_draw
        frame_combine[:,w:,:] = plt_image
        out.write(frame_combine)

        
        # Display the resulting frame
        # Press 'q' to end video
        cv2.imshow('frame',frame_draw)
        cv2.imshow('map',wmap_show)
        cv2.imshow('mask',mask_use)
        cv2.imshow('path',plt_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()







