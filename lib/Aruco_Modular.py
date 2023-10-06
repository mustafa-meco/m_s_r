import cv2 as cv
from cv2 import aruco
import numpy as np

def load_calib_data(path):

    calib_data_path = path

    calib_data = np.load(calib_data_path)
    print(calib_data.files)

    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]

    return cam_mat, dist_coef, r_vectors, t_vectors

def setup_detector(markerSize=4, totalMarkers=50):
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)

    arucoParam = aruco.DetectorParameters()

    return arucoDict, arucoParam

def findArucoMarkers(img, arucoDict, arucoParam, draw=True, convgray=True):

    if convgray:
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        imgGray = img

    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    
    # print(ids)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids, rejected

def setup_camera(device_idx):
    """
    Set up the camera using OpenCV's VideoCapture class
    
    Parameters:
    device_idx (int): index of the camera device
    
    Returns:
    cv2.VideoCapture: a VideoCapture object representing the camera
    """
    cap = cv.VideoCapture(device_idx)
    return cap

def estimate_single_marker_pose(marker_corners, cam_mat, dist_coef, MARKER_SIZE=20):
    rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
    return rVec, tVec

def draw_marker_corners(frame, marker_corners):
    """
    Draws the marker corners on the frame.
    :param frame: (np.array) Input frame.
    :param marker_corners: (List of np.array) Corners of the marker.
    :return: (np.array) Frame with marker corners drawn.
    """
    for corners in marker_corners:
        cv.polylines(
            frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
        )
    return frame

def draw_marker_pose(frame, corners, distance, id):
    cv.polylines(
        frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
    )
    corners = corners.reshape(4, 2)
    corners = corners.astype(int)
    top_right = corners[0].ravel()
    top_left = corners[1].ravel()
    bottom_right = corners[2].ravel()
    bottom_left = corners[3].ravel()

    cv.putText(
        frame,
        f"id: {id} Dist: {round(distance, 2)}",
        top_right,
        cv.FONT_HERSHEY_PLAIN,
        1.3,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )
    

    return frame


if __name__ == '__main__':

    cam_mat, dist_coef, r_vectors, t_vectors = load_calib_data("./calib_data/MultiMatrix.npz")

    MARKER_SIZE = 3.6  # centimeters
    MARKER_BIT_SIZE = 4
    MARKERS_COUNT = 50

    marker_dict, param_markers = setup_detector(MARKER_BIT_SIZE, MARKERS_COUNT)

    cap = setup_camera(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect Aruco markers
        marker_corners, marker_IDs, reject = findArucoMarkers(frame, marker_dict, param_markers)
        if marker_corners:
            rVec, tVec = estimate_single_marker_pose(marker_corners, cam_mat, dist_coef)
            total_markers = range(0, marker_IDs.size)

            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                
                frame = draw_marker_pose()

                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )

                # Draw the pose of the marker
                point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                frame = draw_marker_pose(frame, corners, distance)
                
                # print(ids, "  ", corners)
        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()