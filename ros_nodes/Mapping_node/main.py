import rospy
import rosgraph_msgs.msg as ros_graph_msgs
import sys
import std_msgs.msg as ros_std_msgs
import itertools
import lib.ros as ros_man
import lib.settings as set_man
import lib.Aruco_Modular as AM
import cv2 as cv
from cv2 import aruco
import numpy as np
from lib.mapping import mapping_process
import base64
import pickle
import json

# module config
_NODE_NAME = 'mapping_node'


_processed_feed_pub: rospy.Publisher = None
_car_pos_pub: rospy.Publisher = None
_settings_obj: dict = None


# ros msgs handlers
def _log_read_handler(log: ros_graph_msgs.Log):
    print(f"ROSOUT: {log.msg}")



# change theses in the field
MAID = 1 
MBID = 3
MCID = 7
MDID = 5
MEID = 2
MFID = 5
MGID = 6
MHID = 4

VALID_IDs = [MAID, MBID, MCID, MDID, MEID, MFID, MGID, MHID]

TRACK_WIDTH = 20 # meters
TRACK_HEIGHT= 20 # METERS

MIN_MARKERS_TO_DETECT = 3

ids_cordinate = {MBID: (0, TRACK_HEIGHT), MCID: (TRACK_WIDTH, TRACK_HEIGHT), MDID: (TRACK_WIDTH, 0), MAID: (0, 0), MEID: (0, TRACK_HEIGHT/2), MFID: (TRACK_WIDTH/2, TRACK_HEIGHT), MGID:(TRACK_WIDTH, TRACK_HEIGHT/2), MHID: (TRACK_WIDTH/2, 0)}

def calculate_position(id_list, ids_dist):

    p = [ids_cordinate[id_list[i]] for i in range(id_list)]

    r = [ids_dist[id_list[i]] for i in range(id_list)]
    
    P = [(p[i][0], p[i][1], r[i]) for i in range(id_list)]

    comb = itertools.combinations(P, 3)
    calculations = []
    for c in comb:
        calculations.append(mapping_process(c[0], c[1], c[2]))
    sorted_calculations = sorted(calculations, key=lambda x: x[1])
    pos, err = sorted_calculations[0]
    
    return {'x': pos[0], 'y': pos[1]}

MARKER_SIZE = 28.7  # centimeters
MARKER_BIT_SIZE = 4
MARKERS_COUNT = 50



Car_position = {'x': 0, 'y': 0} # meters

cam_mat, dist_coef, r_vectors, t_vectors = AM.load_calib_data("./calib_data/MultiMatrixBasler.npz")

marker_dict, param_markers = AM.setup_detector(MARKER_BIT_SIZE, MARKERS_COUNT)


Aruco_IDs_dist = {MAID : 0, MBID: 0, MCID: 0, MDID: 0, MEID : 0, MFID: 0, MGID: 0, MHID: 0}
# Aruco_IDs_dist = [0 for _ in range(8)]
detected_Ids_list = []

def _ros_frame_process(msg: ros_std_msgs.String):
    global _processed_feed_pub
    global _car_pos_pub
    
    input_bin_stream = msg.data.encode()

    # base64 decode
    decoded_bin_frame = base64.b64decode(input_bin_stream)

    # recover frame from binary stream
    frame = pickle.loads(decoded_bin_frame)

    # decode JPEG frame
    frame = cv.imdecode(frame, cv.IMREAD_COLOR)

        
    # gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect Aruco markers
    marker_corners, marker_IDs, reject = AM.findArucoMarkers(frame, marker_dict, param_markers, convgray=False)
    if marker_corners:
        rVec, tVec = AM.estimate_single_marker_pose(marker_corners, cam_mat, dist_coef, MARKER_SIZE)
        total_markers = range(0, marker_IDs.size)

        for id, corners, i in zip(marker_IDs, marker_corners, total_markers):
            if id not in VALID_IDs:
                continue
            
            
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            frame = AM.draw_marker_pose(frame, corners, distance, id)

            Aruco_IDs_dist[int(id)] = round(distance / 100, 2) # meters

            detected_Ids_list.append(id)

            detected_Ids_list_unique = np.unique(np.concatenate(detected_Ids_list))

            print(detected_Ids_list_unique)
            if len(detected_Ids_list_unique) >= MIN_MARKERS_TO_DETECT:

                Car_position1 = calculate_position(detected_Ids_list_unique, Aruco_IDs_dist)
                if Car_position1['x'] == None:
                    Car_position = Car_position
                    print('Invalid_Coordinates -- NO CIRCLES INTERSECTION')
                elif Car_position1['x'] > TRACK_WIDTH or Car_position1['y'] > TRACK_HEIGHT or Car_position1['x'] < 0 or Car_position1['y'] < 0:
                    Car_position = Car_position
                    print('Invalid_Coordinates -- OUT OF TRACK')
                else:
                    Car_position = Car_position1
                    print('Car_Position:',Car_position)
                detected_Ids_list = []
            else:
                Car_position = Car_position

    
            
            # print(ids, "  ", corners)
    
    _car_pos_pub.publish(json.dumps(Car_position))

    processed_frame = cv.resize(frame, (400, 400))
    _, compressed_frame = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 90])

    # convert frame to binary data
    bin_frame = pickle.dumps(compressed_frame, pickle.HIGHEST_PROTOCOL)

    # base64 encode frame
    encoded_bin_frame = base64.b64encode(bin_frame).decode()

    # publish frame in ROS
    _processed_feed_pub.publish(encoded_bin_frame)
    


def ros_node_setup():
    global _processed_feed_pub
    global _car_pos_pub
    global _settings_obj

    is_init = ros_man.init_node(_NODE_NAME)
    
    if not is_init:
        sys.exit()

    
    
    _settings_obj = set_man.get_settings()

    feed_processed_topic_id = ros_man.create_topic_id('basler_feed_processed')
    q_size: int = _settings_obj['ros']['msg_queue_size']

    _processed_feed_pub = rospy.Publisher(
        feed_processed_topic_id, ros_std_msgs.String, queue_size=q_size)
    
    pos_topic_id = ros_man.create_topic_id('car_pos')
    _car_pos_pub = rospy.Publisher(
        pos_topic_id, ros_std_msgs.String, queue_size=q_size)

    
    rospy.Subscriber('/basler_adapter_node/basler_feed_normal', ros_graph_msgs.Log, _ros_frame_process)
    



def ros_node_loop():
    pass