import matplotlib.pyplot as plt
import os, sys, glob
from natsort import natsorted
from scipy.stats  import gaussian_kde
import numpy as np
import cv2
import math
import pandas as pd
from PIL import Image, ImageOps

class SMI_calculations:
    def __init__(self, data_path):
        self.step = 0.01  # gradient step
        self.delta = 0.000001  # gradient break value
        self.pi2 = np.pi / 2
        self.m = 1

        self.data_path = data_path
        self.calib_path = self.data_path / 'results' / 'calib'
        self.image_data_path = self.data_path / 'results' / 'leftImage' / 'data'
        self.lidar_data_path = self.data_path / 'results' / 'velodyne_points' / 'data'

        self.image_files = [Image.open(x) for x in self.image_data_path.glob('*.bmp') if x.is_file()]
        self.lidar_data = [pd.read_csv(x) for x in self.lidar_data_path.glob('*.csv') if x.is_file()]

        self.K, self.D = self.read_calib_data()

    def read_calib_data(self):
        cam_mono = cv2.FileStorage(self.calib_path + "/cam_mono.yml", cv2.FILE_STORAGE_READ)
        K = cam_mono.getNode("K")
        D = cam_mono.getNode("D")
        K_final = np.array(K.mat())
        D_final = np.array(D.mat())
        K_final[0, 0] = 2 * K_final[0, 0]
        K_final[1, 1] = 2 * K_final[1, 1]
        D_final = [x[0] for x in D_final]
        return (K_final, D_final)

    def get_intensivity(pixel, img):
        img = img.convert('L')
        x = int(pixel[0] + 959)
        y = int(pixel[1] + 539)
        return img.getpixel((x, y))

    def calc_SMI(self, points, Lidar_data_file, image):
        SMI = 0.0
        list_ref = []
        list_intens = []
        for i in range(len(Lidar_data_file)):
            pixel = self.Projection(Lidar_data_file[i][:3], points, self.K, self.D)
            if pixel is not None:
                # print ("pixel ",pixel[:2])
                list_ref.append(Lidar_data_file[i][3])
                list_intens.append(self.get_intensivity(pixel[:2], image))
                # print("list_intens",list_intens)
        # print("list_ref",list_ref)
        kernel_r = gaussian_kde(list_ref)
        ref = kernel_r.evaluate(range(0, 255))
        kernel_i = gaussian_kde(list_intens)
        inte = kernel_i.evaluate(range(0, 255))
        mutual = np.histogram2d(list_ref, list_intens, bins=255, range=[[0, 255], [0, 255]], density=True)
        for i in range(0, 255):
            for j in range(0, 255):
                SMI += 0.5 * ref[i] * inte[j] * ((mutual[0][i][j] / (ref[i] * inte[j])) - 1) ** 2
        return (SMI)

    def read_scv(self, ):
        SMI_point = [0, 0, 0, 0, 0, 0]
        cur_points = [-self.pi2, 0, 2 * self.pi2, -0.885, -0.066, 0]
        average_points = [-1 * self.pi2, 0, 2 * self.pi2, -0.885, -0.066, 0]
        cur_SMI = self.calc_SMI(cur_points, self.lidar_data[0], self.image_files[0])
        cur_gradient = self.calculate_gradient(cur_points, self.lidar_data[0], self.image_files[0])

        for i in range(len(self.lidar_data)):
            print("SMI calc for ", i, "file")
            next_gradient = cur_points + cur_gradient * self.step * 10
            next_SMI = self.calc_SMI(next_gradient, self.lidar_data[i], self.image_files[i])

            if next_SMI > cur_SMI:

                if next_SMI < cur_SMI + self.delta:
                    break

                cur_SMI = next_SMI
                cur_points = next_gradient
                self.m += 1
                for k in range(len(cur_points)):
                    average_points[k] += cur_points[k]
            else:
                cur_gradient = self.calculate_gradient(cur_points,
                                self.lidar_data[i], self.image_files[i])

        SMI_point = [SMI_point[i] + average_points[i] / self.m for i in range(len(average_points))]
        print(SMI_point)

        # Visual part after calibration
        for i in range(len(self.lidar_data)):
            print("plot for ", i, "file after calib")
            dataset = self.lidar_data[i]
            pixels = []
            for j in range(len(self.lidar_data[i])):
                pixel = self.Projection(dataset[j][:3], SMI_point)
                if pixel is not None:
                    pixels.append(pixel)
            df = pd.DataFrame(pixels, columns=['x', 'y', 'z'])
            fig = self.dfScatter(i, df)
            fig.savefig(str(i) + '_after.png', dpi=60)

    def Projection(self, Lidar_data_line, cam_pos, K, D):
        # Rotation
        roll = [[1, 0, 0], [0, np.cos(cam_pos[0]), -np.sin(cam_pos[0])],
                [0, np.sin(cam_pos[0]), np.cos(cam_pos[0])]]
        pitch = [[np.cos(cam_pos[1]), 0, np.sin(cam_pos[1])], [0, 1, 0],
                 [-np.sin(cam_pos[1]), 0, np.cos(cam_pos[1])]]
        yaw = [[np.cos(cam_pos[2]), -np.sin(cam_pos[2]), 0],
               [np.sin(cam_pos[2]), np.cos(cam_pos[2]), 0], [0, 0, 1]]
        R = np.dot(roll, np.dot(pitch, yaw))
        rotation = np.dot(R, Lidar_data_line[:3])
        # Translation
        cam_coord = [rotation[0] + cam_pos[3], rotation[1] + cam_pos[4], rotation[2] + cam_pos[5]]
        # Projection
        w = cam_coord[2]
        v = cam_coord[1]
        cam_coord = cam_coord / w
        P = np.zeros([3, 3])
        P[:, :] = K[:, :]
        pixels = np.dot(P, cam_coord)

        if (abs(pixels[0]) < 960) and (abs(pixels[1]) < 540):
            d = np.sqrt(sum(i * i for i in cam_coord))
            pixels[2] = d
            """
            #Distor
            r=cam_coord[0]**2+cam_coord[1]**2
            Tan=math.atan(r)
            cam_coord[0]=(1+D[0]*r+D[1]*(r**2)+D[4]*(r**3))*cam_coord[0]*Tan/r
            cam_coord[1]=(1+D[0]*r+D[1]*(r**2)+D[4]*(r**3))*cam_coord[1]*Tan/r
            P = np.zeros([3, 3])
            P[:, :] = K[:, :]
            pixels = np.dot(P, cam_coord)
            pixels[2] = v"""
            return (pixels)

    def dfScatter(self, i, df, xcol='x', ycol='y', catcol='z'):
        fig, ax = plt.subplots(figsize=(20, 10), dpi=60, )
        categories = np.unique(df[catcol])
        colors = np.linspace(categories.min(), categories.max(), len(categories))
        colordict = dict(zip(categories, colors))
        df["c"] = df[catcol].apply(lambda k: colordict[k])
        img = ImageOps.mirror(self.image_files[i])
        sc = plt.scatter(df[xcol], df[ycol], c=df.c, zorder=2, s=10)
        plt.imshow(img, extent=[df[xcol].min(), df[xcol].max(), df[ycol].min(), df[ycol].max()], zorder=0,
                   aspect='auto')
        colorize = plt.colorbar(sc, orientation="horizontal")
        colorize.set_label("distance (m)")
        return fig

    def calculate_gradient(self,points, Lidar_data_file, image):
        # points = [alpha, beta, gamma, u0, v0, w0]
        gradient = np.zeros(6)
        for i in range(len(points)):
            up_points = points
            down_points = points
            up_points[i] += self.step
            down_points[i] -= self.step
            gradient[i] = (self.calc_SMI(up_points, Lidar_data_file, image) - self.calc_SMI(down_points,
                                                                                        Lidar_data_file, image)) / (2 * step)
        return (gradient)


