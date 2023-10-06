import cv2
from pypylon import pylon

class BaslerCamera:
    def __init__(self, exposure_time=None):
        # Create a camera object
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # Start grabbing images
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        # Set the exposure time if provided
        if exposure_time is not None:
            self.camera.ExposureTime.SetValue(exposure_time)
    
    def get_frame(self):
        # Get the latest image
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # Check if the image is valid
        if grabResult.GrabSucceeded():
            # Access the image data
            image = grabResult.Array
            # Convert the image from BGR to RGB format
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            return None

    def stop_capture(self):
        # Stop grabbing images
        self.camera.StopGrabbing()

