#!/usr/bin/env python
import rospy
import cv2
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist

global counter
global count_max
counter = 0
count_max = 400

global fourcc
global out
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #Define the codec and create VideoWriter object
out = cv2.VideoWriter('/home/splinter/output_hog.avi',fourcc, 20.0, (640,480))

class CVControl:

    def __init__(self):
    
        # Turtlebot command publisher
        self.cmd_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)
        self.cmd = Twist()

        # Image subscriber
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/decompressed_img", Image, self.img_callback)

    def img_callback(self, data):
    global counter
    counter += 1    
    
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e
            
     # Get grayscale image       
    cv_gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
    

    # detect people in the image for every 2nd frame
    if counter%2 == 0:
        # detect humans 
        (rects, weights) = hog.detectMultiScale(cv_gray, winStride=(8, 8), padding=(8, 8),scale=1.03)
        [x1,y1,w1,h1] = [0,0,480,640] 
        if len(weights) > 0: # if person detected
            idx = np.argmax(weights) # choose person detected with highest probability
            if weights[idx] > 1.2: # if high probability it's a person
                # find and draw rectangle around person
                rect = rects[idx,:]
                [x1,y1,w1,h1] = rect
                cv2.rectangle(cv_image,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
                # find rectangel area and horizontal center
                rect_center = int(x1+w1/2)
                rect_area = w1*h1
                if rect_area > 25000:
                    # use proportional control to center rectangle with target area size
                    img_center = 320
                    target_area = 100000
                    kr = .002
                    w = -kr*(rect_center-img_center)
                    kt = 0.0000045
                    v = kt*(target_area-rect_area)
                    # limit to a max velocity
                    maxv = 0.25
                    v = np.max([-maxv, v])
                    v = np.min([maxv, v])
                    # Send Velocity command to turtlebot
                    self.send_command(v, w)    

    if counter < count_max:
        out.write(cv_image)
        print(counter)
    if counter == count_max:
        out.release()
        print('made video')
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

    def send_command(self, v, w):
        # Put v, w commands into Twist message
        self.cmd.linear.x = v
        self.cmd.angular.z = w

        # Publish Twist command
        self.cmd_pub.publish(self.cmd)

def main():
    ctrl = CVControl()
    rospy.init_node('image_converter')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print(cv2.__version__)
    main()
