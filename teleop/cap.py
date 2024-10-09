# import pyzed.sl as sl
# import cv2

# # Create a Camera object
# zed = sl.Camera()

# init_params = sl.InitParameters()
# init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
# init_params.camera_fps = 30

# # Open the camera
# status = zed.open(init_params)

# if status != sl.ERROR_CODE.SUCCESS:
#     print(f"Error opening camera: {status}")
#     exit(1)

# # Grab an image
# runtime_parameters = sl.RuntimeParameters()

# i = 0
# image = sl.Mat()
# runtime_parameters = sl.RuntimeParameters()
# while i < 1000:
# # Grab an image, a RuntimeParameters object must be given to grab()
#     if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
#         # A new image is available if grab() returns ERROR_CODE.SUCCESS
#         zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
#         # Use get_data() to get the numpy array
#         image_zed = sl.Mat(zed.get_camera_information().camera_configuration.resolution.width, zed.get_camera_information().camera_configuration.resolution.height, sl.MAT_TYPE.U8_C4)
#         image_ocv = image_zed.get_data()
#          # Display the left image from the numpy array
#         cv2.imshow("Image", image_ocv)
#         # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp
#         # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(), timestamp.get_milliseconds()))
#         i = i + 1
# # Close the camera
# zed.close()

import cv2
import numpy

# Open the ZED camera
cap = cv2.VideoCapture(0)
if cap.isOpened() == 0:
    exit(-1)

# Set the video resolution to HD720 (2560*720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True :
    # Get a new frame from camera
    retval, frame = cap.read()
    # Extract left and right images from side-by-side
    left_right_image = numpy.split(frame, 2, axis=1)
    # Display images
    cv2.imshow("frame", frame)
    cv2.imshow("right", left_right_image[0])
    cv2.imshow("left", left_right_image[1])
    if cv2.waitKey(30) >= 0 :
        break

exit(0)