from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import pandas as pd
from PIL import Image, ImageOps


class Camera:

    def __init__(self, data_path):
        self.pi2 = np.pi / 2
        self.camera_rotation_angles = [-self.pi2, 0, 2 * self.pi2]
        self.data_path = data_path
        self.calib_path = self.data_path / 'calib'
        self.image_data_path = self.data_path / 'leftImage' / 'data'
        self.image_files = [x for x in self.image_data_path.glob('*.bmp') if x.is_file()]
        self.K, self.D = self.read_calib_data()

    def read_calib_data(self):
        cam_mono = cv2.FileStorage(str(self.calib_path/'cam_mono.yml'), cv2.FILE_STORAGE_READ)
        K = cam_mono.getNode("K").mat()
        D = cam_mono.getNode("D").mat()
        K_final = np.array(K)
        D_final = np.array(D)
        K_final[0, 0] = 2 * K_final[0, 0]
        K_final[1, 1] = 2 * K_final[1, 1]
        D_final = [x[0] for x in D_final]
        return K_final, D_final

    def translation_pts_to_cam_sys(self, rotated_points, lidar_pos_in_cam_sys, cam_angles = None):
        if cam_angles is None:
            cam_coord = [rotated_points[0] + lidar_pos_in_cam_sys[0],
                         rotated_points[1] + lidar_pos_in_cam_sys[1],
                         rotated_points[2] + lidar_pos_in_cam_sys[2]]
        else:
            cam_coord = [rotated_points[0] + cam_angles[0],
                         rotated_points[1] + cam_angles[1],
                         rotated_points[2] + cam_angles[2]]
        return cam_coord

    def projection_pts_on_camera(self, translated_points):
        w = translated_points[2]
        if w != 0:
            translated_points = translated_points / w
            pixel = np.dot(self.K, translated_points)

            if (abs(pixel[0]) < 960) and (abs(pixel[1]) < 540):
                r = translated_points[0] ** 2 + translated_points[1] ** 2
                Tan = math.atan(r)
                translated_points[0] = (1 + self.D[0] * r + self.D[1] * (r ** 2) + self.D[4] * (r ** 3)) * \
                                       translated_points[0] * Tan / r
                translated_points[1] = (1 + self.D[0] * r + self.D[1] * (r ** 2) + self.D[4] * (r ** 3)) * \
                                       translated_points[1] * Tan / r
                pixel = np.dot(self.K, translated_points)
                return pixel


class Lidar():

    def __init__(self, data_path, camera):
        # lidar position in cam coord
        self.lidar_pos_in_cam_sys = [-0.885, -0.066, 0]

        self.data_path = data_path
        self.lidar_data_path = self.data_path / 'velodyne_points' / 'data'
        self.lidar_data = [x for x in self.lidar_data_path.glob('*.csv') if x.is_file()]
        self.local_cam = camera

    def projection_pts_on_cam(self, csv_data, cam_point = None):
        #print("scv ", csv_data)
        #print("points", cam_point)
        if cam_point is None:
            alpha = self.local_cam.camera_rotation_angles[0]
            betta = self.local_cam.camera_rotation_angles[1]
            gamma = self.local_cam.camera_rotation_angles[2]
            cam_point = [alpha, betta, gamma, self.lidar_pos_in_cam_sys]
        else:
            alpha = cam_point[0]
            betta = cam_point[1]
            gamma = cam_point[2]

        roll = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]]
        pitch = [[np.cos(betta), 0, np.sin(betta)], [0, 1, 0],
                 [-np.sin(betta), 0, np.cos(betta)]]
        yaw = [[np.cos(gamma), -np.sin(gamma), 0],
               [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]
        r_matrix = np.dot(roll, np.dot(pitch, yaw))

        rotated_to_cam_points = np.dot(r_matrix, csv_data)
        #print("after rotation ", rotated_to_cam_points)

        translated_to_cam_pts = self.local_cam.translation_pts_to_cam_sys(rotated_to_cam_points, cam_point[3:], self.lidar_pos_in_cam_sys)
        #print("after translations", translated_to_cam_pts)

        point_height = translated_to_cam_pts[1]
        point_distance = np.sqrt(sum(i * i for i in translated_to_cam_pts))

        projected_to_camera_points = self.local_cam.projection_pts_on_camera(translated_to_cam_pts)
        #print("after projections", projected_to_camera_points)

        return projected_to_camera_points, point_height, point_distance

    def dfScatter(self, i, df, xcol ='x', ycol ='y', catcol ='color'):
        fig, ax = plt.subplots(figsize=(20, 10), dpi=60, )
        categories = np.unique(df[catcol])
        colors = np.linspace(categories.min(), categories.max(), len(categories))
        colordict = dict(zip(categories, colors))
        df["c"] = df[catcol].apply(lambda k: colordict[k])
        img = ImageOps.mirror(Image.open((self.image_files[i])))
        sc = plt.scatter(df[xcol], df[ycol], c=df.c, zorder=2, s=10)
        plt.imshow(img, extent=[df[xcol].min(), df[xcol].max(), df[ycol].min(), df[ycol].max()], zorder=0, aspect='auto')
        colorize = plt.colorbar(sc, orientation="horizontal")
        colorize.set_label("distance (m)")
        return fig

class SMI_calculations:
    def __init__(self, data_path):
        self.step = 0.01         # gradient step
        self.delta = 0.000001    # gradient break value
        self.pi2 = np.pi / 2
        self.m = 1

        self.local_camera = Camera(data_path)
        self.local_lidar = Lidar(data_path, self.local_camera)

    def power(self):
        SMI = self.SMI_point_calculations()
        self.SMI_Visualization(SMI, color = "Distance")

    def SMI_point_calculations(self, ):

        SMI_point = [0, 0, 0, 0, 0, 0]
        cur_points = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
        average_points = cur_points

        cur_SMI = self.calc_SMI(cur_points, self.local_lidar.lidar_data[0], self.local_camera.image_files[0])
        cur_gradient = self.calculate_gradient(cur_points, self.local_lidar.lidar_data[0],
                                               self.local_camera.image_files[0])
        #print(cur_SMI)

        for i in range(1, len(self.local_camera.image_files) - 1):
            print("SMI calc for ", i, "file")
            next_gradient = cur_points + cur_gradient * self.step * 10
            next_SMI = self.calc_SMI(next_gradient, self.local_lidar.lidar_data[i], self.local_camera.image_files[i])

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
                                                       self.local_lidar.lidar_data[i], self.local_camera.image_files[i])

        SMI_point = [SMI_point[i] + average_points[i] / self.m for i in range(len(average_points))]

        return SMI_point

    def calc_SMI(self, points, Lidar_file, image):
        Lidar_data = pd.read_csv(Lidar_file, header=None)
        SMI = 0.0
        list_ref = []
        list_intens = []

        for i in range(len(Lidar_data)):
            pixel, h, d = self.local_lidar.projection_pts_on_cam(Lidar_data.iloc[i][0:3], points)
            if pixel is not None:
                list_ref.append(Lidar_data.iloc[i][3])
                list_intens.append(self.get_intensivity(pixel[:2], image))
        #print("list_intens",list_intens)
        #print("list_ref",list_ref)
        if list_ref:
            kernel_r = gaussian_kde(list_ref)
            ref = kernel_r.evaluate(range(0, 255))
            kernel_i = gaussian_kde(list_intens)
            inte = kernel_i.evaluate(range(0, 255))
            mutual = np.histogram2d(list_ref, list_intens, bins=255, range=[[0, 255], [0, 255]], density=True)
            for i in range(0, 255):
                for j in range(0, 255):
                    SMI += 0.5 * ref[i] * inte[j] * ((mutual[0][i][j] / (ref[i] * inte[j])) - 1) ** 2
        return SMI

    def get_intensivity(self, pixel, img):
        img = Image.open(img).convert('L')
        x = int(pixel[0] + 959)
        y = int(pixel[1] + 539)
        return img.getpixel((x, y))

    def calculate_gradient(self, points, Lidar_file, image):
        # points = [alpha, beta, gamma, u0, v0, w0]
        gradient = np.zeros(6)
        for i in range(len(points)):
            up_points = points
            down_points = points
            up_points[i] += self.step
            down_points[i] -= self.step
            gradient[i] = (self.calc_SMI(up_points, Lidar_file, image) - self.calc_SMI(down_points,
                                                    Lidar_file, image)) / (2 * self.step)
        return gradient

    def SMI_Visualization(self, SMI_point, color = "Height"):
        for i in range(len(self.local_lidar.lidar_data)):
            print("plot for ", i, "file after calib,", "SMI_point =", SMI_point, "color = ", color)
            dataset  = pd.read_csv(self.local_lidar.lidar_data[i], header=None)
            pixels = []
            heights = []
            distances = []

            for j in range(len(dataset)):
                pixel, height, distance = self.local_lidar.projection_pts_on_cam(dataset.iloc[j][0:3], SMI_point)
                if (pixel[0] and pixel[1]) is not None:
                    pixels.append(pixel)
                    heights.append(height)
                    distances.append(distance)

            if color == "Height":
                df = pd.DataFrame(pixels[0:1], columns=['x', 'y'])
                df.insert(2, "color", heights, True)

            else:
                df = pd.DataFrame(pixels[0:1], columns=['x', 'y'])
                df.insert(2, "color", distances, True)

            fig = self.local_lidar.dfScatter(i, df)

            fig.savefig(str(i) + '_at_SMI.png', dpi=60)