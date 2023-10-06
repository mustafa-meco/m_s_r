import rospy
import rosgraph_msgs.msg as ros_graph_msgs
import sys
import std_msgs.msg as ros_std_msgs
import lib.ros as ros_man
import lib.settings as set_man
import numpy as np
import base64
import pickle
import json
import cv2
import pickle
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

# module config
_NODE_NAME = 'mine_vision_node'


_mine_processed_feed_pub: rospy.Publisher = None
_upper_flag_pub: rospy.Publisher = None
_settings_obj: dict = None


# ros msgs handlers
def _log_read_handler(log: ros_graph_msgs.Log):
    print(f"ROSOUT: {log.msg}")


mine_length=70
mine_area=mine_length**2
bilateral_factor=105
thresh=128
img_len=400
#sobel_ksize=5
#gaussian_size=55
#median_size=75



def _ros_mine_frame_process(msg: ros_std_msgs.String):
    global _mine_processed_feed_pub
    global _upper_flag_pub
    
    input_bin_stream = msg.data.encode()

    # base64 decode
    decoded_bin_frame = base64.b64decode(input_bin_stream)

    # recover frame from binary stream
    frame = pickle.loads(decoded_bin_frame)

    # decode JPEG frame
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    bilateral = cv2.bilateralFilter(frame, bilateral_factor, img_len, img_len)

    img_gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    _, img_threshold = cv2.threshold(img_gray, thresh, 1, cv2.THRESH_BINARY_INV)

    upper_flag=False
    all_sum=img_threshold.sum()
    upper_flag=True if all_sum>mine_area else 0

    
    _upper_flag_pub.publish(json.dumps(upper_flag))

    if upper_flag==1:
        cols = img_threshold.sum(axis=0) #sum on rows ==> gives columns weight
        rows = img_threshold.sum(axis=1) #sum on columns ==> gives rows weight
        x = rows.argsort()[::-1][0] # Getting index of the maximum row value
        y = cols.argsort()[::-1][0] # Getting index of the maximum column value
        mine_index=img_threshold[x][y] #best pixel in the image that represents and refer to the mine, Must = 1                                       

        _, img_threshold = cv2.threshold(img_gray, thresh, 1, cv2.THRESH_BINARY_INV)

        canny = cv2.Canny(img_threshold,0,1)

        x=signal.convolve2d(canny, np.ones((5,5))*(1/25), boundary='symm', mode='same')

        colored_frame=np.zeros((400,400,3))
        colored_frame[:,:,0]=x
        colored_frame[:,:,1]=img_threshold
        frame = colored_frame
    

    processed_frame = cv2.resize(frame, (400, 400))
    _, compressed_frame = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # convert frame to binary data
    bin_frame = pickle.dumps(compressed_frame, pickle.HIGHEST_PROTOCOL)

    # base64 encode frame
    encoded_bin_frame = base64.b64encode(bin_frame).decode()

    # publish frame in ROS
    _mine_processed_feed_pub.publish(encoded_bin_frame)
    


def ros_node_setup():
    global _mine_processed_feed_pub
    global _upper_flag_pub
    global _settings_obj

    is_init = ros_man.init_node(_NODE_NAME)
    
    if not is_init:
        sys.exit()

    
    
    _settings_obj = set_man.get_settings()

    mine_feed_processed_topic_id = ros_man.create_topic_id('mine_feed_processed')
    q_size: int = _settings_obj['ros']['msg_queue_size']

    _processed_feed_pub = rospy.Publisher(
        mine_feed_processed_topic_id, ros_std_msgs.String, queue_size=q_size)
    
    upper_flag_topic_id = ros_man.create_topic_id('upper_flag')
    _upper_flag_pub = rospy.Publisher(
        upper_flag_topic_id, ros_std_msgs.String, queue_size=q_size)

    
    rospy.Subscriber('/camera_adapter_node/camera_feed', ros_graph_msgs.Log, _ros_mine_frame_process)
    



def ros_node_loop():
    pass