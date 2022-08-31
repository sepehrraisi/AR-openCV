
__author__ = 'Sepehr'

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np


class CamApp(App):

    def build(self):
        self.stdShape = (275, 440)
        self.webCam = cv2.VideoCapture(0)
        self.imgTarget = cv2.imread("assets/1.jpg")
        self.imgTarget = cv2.resize(self.imgTarget, self.stdShape)
        # displayVid = cv2.VideoCapture("assets\displayVid.mp4")
        self.displayImg = cv2.imread("assets/2.jpg")
        self.ORB = cv2.ORB_create(nfeatures=1000)
        self.keyPoint1, self.descriptor1 = self.ORB.detectAndCompute(self.imgTarget, None)
        self.img1 = Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        # opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    def update(self, dt):

        _, imgWebcam = self.capture.read()
        keyPoint2, descriptor2 = self.ORB.detectAndCompute(imgWebcam, None)

        imgAR = imgWebcam.copy()

        # _ , imgVideo = displayVid.read()
        # imgVideo = cv2.resize(imgVideo, stdShape)
        imgVideo = cv2.resize(self.displayImg, self.stdShape)

        bruteForce = cv2.BFMatcher()
        matches = bruteForce.knnMatch(self.descriptor1, descriptor2, k=2)

        goodMatches = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatches.append(m)

        if len(goodMatches) > 15:
            srcPts = np.float32([self.keyPoint1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            dstPts = np.float32([keyPoint2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

            pts = np.float32([[0, 0], [0, 440], [275, 440], [275, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            cv2.polylines(imgWebcam, [np.int32(dst)], True, (0, 0, 255), 3)
            # cv2.imshow("Polylines", imgWebcam)
            imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

            newmask = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
            cv2.fillPoly(newmask, [np.int32(dst)], (255, 255, 255))

            invMask = cv2.bitwise_not(newmask)
            imgAR = cv2.bitwise_and(imgAR, imgAR, mask=invMask)
            imgAR = cv2.bitwise_or(imgAR, imgWarp)

        # cv2.imshow("imgAR", imgAR)
        buf1 = cv2.flip(imgAR, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(imgAR.shape[1], imgAR.shape[0]), colorfmt='bgr')
        # if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()
