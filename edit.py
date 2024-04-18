import numpy as np
import cv2
import sys
import mediapipe as mp
import math
import time

mp_pose = mp.solutions.pose 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
def reconstructFrame(pyramid, index, levels, videoHeight, videoWidth):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

def calculate_color(bpm):
    if bpm < 70:
        return (0, 0, 0)  # Black for BPM less than 70
    elif bpm >= 70 and bpm < 80:
        return (75, 0, 130)  # Indigo for BPM 70-79
    elif bpm >= 80 and bpm < 90:
        return (138, 43, 226)  # Violet for BPM 80-89
    elif bpm >= 90 and bpm < 100:
        return (0, 0, 255)  # Blue for BPM 90-99
    elif bpm >= 100 and bpm < 110:
        return (0, 255, 0)  # Green for BPM 100-109
    elif bpm >= 110 and bpm < 120:
        return (255, 255, 0)  # Yellow for BPM 110-119
    elif bpm >= 120 and bpm < 130:
        return (255, 127, 0)  # Orange for BPM 120-129
    else:
        return (255, 0, 0)  # Red for BPM 130 and above

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

#Variables
videoHeight = 120
videoWidth = 160
image_depth = 0
videoChannels = 3
videoFrameRate = 15

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (videoWidth-20, 100)
fontScale = 2
fontColor = (0,0,0)
fontThickness = 3
shadowColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3
timeTextLocation = (1500,100)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

#Bandpass Filter
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

#MediaPipe Code
cap = cv2.VideoCapture(0)
videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
bpmTextLocation = (videoWidth-800, 100)

#Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

startTime = time.time()
cv2.namedWindow("MediaPipe Pose")
i=0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    videoHeight, videoWidth, image_depth = image.shape
    if not results.pose_landmarks:
      continue
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    currentTime = time.time()
    elapsedTime = currentTime - startTime
    
    # Pink header!
    headerBackgroundColor = (193, 182, 255)
    headerHeight = 150
    cv2.rectangle(image, (0, 0), (videoWidth, headerHeight), headerBackgroundColor, -1)
    headerTextLocation = (10, headerHeight - 15)
    cv2.putText(image, "How long can you go without blinking?", loadingTextLocation, font, fontScale, fontColor, fontThickness, lineType)

    if elapsedTime < 3:
        #Landmark Coords & Frame Calculation
        noseCoords_X = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * videoWidth
        noseCoords_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * videoHeight
        noseCoords_Z = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z * image_depth

        leftEyeInner_X = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].x * videoWidth
        rightEyeInner_X = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].x * videoWidth

        leftEyeInner_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].y * videoHeight
        rightEyeInner_Y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y * videoHeight
        averageEye_Y = (leftEyeInner_Y + rightEyeInner_Y)/2
        noseBridge_length = averageEye_Y - noseCoords_Y

        x1 = math.floor(leftEyeInner_X)
        y1 = math.floor(averageEye_Y - noseBridge_length)
        x2 = math.floor(rightEyeInner_X)
        y2 = math.floor(averageEye_Y)
        print(x1,y1,x2,y2)
        # TIMER
        timerSeconds = int(4-elapsedTime)
        timerLocation = (900, 600)
        timerFontScale = 6
        textThickness = 10
        cv2.putText(image, f"{timerSeconds}", timerLocation, font, timerFontScale, (0, 0, 255), textThickness, lineType)

    else:
        elapsedSeconds = int(elapsedTime -3)
        cv2.putText(image, f"{elapsedSeconds} s", timeTextLocation, font, fontScale, fontColor, lineType)
        #HEARTRATE CALCULATION
        #Construct ROI
        roi1 = image[min(x1,x2):max(x1,x2), min(y1,y2):max(y1,y2), :]
        roi1_resized = cv2.resize(roi1, (240, 135))  # Resize ROI

        # Build Gaussian Pyramid
        gauss_pyramid = buildGauss(roi1_resized, levels+1)
        desired_level = gauss_pyramid[levels]
        desired_level_resized = cv2.resize(desired_level, (240, 135))  # Ensuring size matches videoGauss
        videoGauss[bufferIndex] = desired_level_resized
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha
        
        bufferIndex = (bufferIndex + 1) % bufferSize
        
        #Output BPM
        if i > bpmBufferSize:
            # cv2.putText(image, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
            color = calculate_color(bpmBuffer.mean())
            # Define circle parameters
            circle_center = (200, 900)  # Top-left corner for demonstration, adjust as needed
            circle_radius = 150
            # Draw the filled circle on the frame
            cv2.circle(image, circle_center, circle_radius, color, -1)  # -1 thickness makes the circle filled
            bpmTextLocation = (60,920)
            bpmFontScale = 1.8
            cv2.putText(image, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, bpmFontScale, (255,255,255), lineType)

    # Exit on ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break
    cv2.rectangle(image, (x1, y1), (x2, y2), boxColor, boxWeight)
    cv2.imshow('MediaPipe Pose', image)

cap.release()
cv2.destroyAllWindows()