import numpy as np
import pandas as pd
import re
import sys
import struct
from natsort import natsorted
import gzip
from datetime import datetime
from distutils.dir_util import copy_tree
import yaml
import shutil
from distutils import dir_util
import calendar
from pathlib import Path
import shutil


LASER_ANGLES = [-15, 1, -13, -3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
NUM_LASERS = 16
DISTANCE_RESOLUTION = 0.002
ROTATION_MAX_UNITS = 36000


class Engine:
    def __init__(self, data_path):
        self.path = Path(data_path) / 'data'
        self.pcap_path = self.path / 'pcap'
        self.yml_path = self.path / 'yml'
        self.images = self.path / 'frames'
        self.pcap_files = [x for x in self.pcap_path.glob('*.*') if x.is_file()]
        self.yml_files = [x for x in self.yml_path.glob('*.*') if x.is_file()]
        self.image_files = [x for x in self.images.glob('*.bmp') if x.is_file()]

        self.results = Path(data_path) / 'results'
        if self.results.is_dir():
            shutil.rmtree(str(self.results))
        self.results.mkdir()

        self.image_names_done = []
        self.VideoFlows = []
        self.VideoNumbers = []
        self.counter = 0
        self.time_lidar = None
        self.leftImage_grabMsec = None
        self.leftImage_deviceSec = None
        self.pacTimeStamps = None
        self.dataset = {'X': [], 'Y': [], 'Z': [], 'ref': [], 'azimuth': [], 'laser_id': []}
        self.XYZD_info = pd.DataFrame(self.dataset)

    def power(self, step):
        yml_file = str(self.yml_files[step])
        file_name = (yml_file.split('\\')[-1])[:-3]
        print(file_name)
        self.regular_expression(yml_file=file_name)
        self.read_yml(filename=yml_file)

    def regular_expression(self, yml_file):
        mat = re.match(r"(?P<flow>\S+)\.(?P<VideoFlow>\d+)\.(?P<VideoNumber>\d+)\.(?P<info>\S+)\.(?P<type>\d*)",
                       yml_file)
        for key, val in (mat.groupdict()).items():
            if key.startswith("VideoNumber"):
                self.VideoNumbers.append(val)
            if key.startswith("VideoFlow"):
                self.VideoFlows.append(val)
                # out = os.path.join("Kitti", VideoFlows[-1])

    def read_yml(self, filename):
        with gzip.open(filename, "rt") as file:
            #data = yaml.safe_load_all(config)
            config = file.read()
            image_number = 0
            if config.startswith("%YAML:1.0"):
                config = "%YAML 1.1" + str(config[len("%YAML:1.0"):])
                data = list(yaml.load_all(config))
                for shot in data[0]['shots']:
                    self.shot_prcessing(image_number=image_number, shot=shot)

    def shot_prcessing(self, image_number, shot):
        """read lidar timestamps data """
        for key, value in sorted(shot.items(), reverse=True):
            if key.startswith("velodyneLidar"):
                self.timestamps_processing(shot, key, value)

            """read camera timestamps and check is frame name in (good frames_dir)"""
            if key.startswith("leftImage"):
                self.camera_timestamps_processing(shot, key, value, image_number)

                """move images from frame folder to Kitti struct with time from yaml_file"""
                if (YAML_Image_names in Image_names) and (YAML_Image_names not in Image_names_done):
                    Image_names_done.append(YAML_Image_names)
                    move_images(self.counter, frames_dir, YAML_Image_names, out,
                                self.leftImage_grabMsec / 1e6 + self.leftImage_deviceSec)
                    time_lidar = datetime.fromtimestamp(
                        self.leftImage_grabMsec / 1e6 + self.leftImage_deviceSec).strftime('%Y-%m-%d_%H_%M_%S.%f')
                    # print ("save  time ",leftImage_grabMsec)
                    save_lidar_data_time(time_lidar, out, self.counter, Xself.YZD_info)
                    # XYZD_info = pd.DataFrame(dataset)
                    self.counter += 1

    def lidar_timestamps_processing(self, shot, key, value):
        if key.startswith("velodyneLidar"):
            velodyneLidars = shot['velodyneLidar']
            for velo_key, velo_value in velodyneLidars.items():
                if velo_key.startswith("lidarData"):
                    pacTimeStamps = velo_value["pacTimeStamps"]
                    print("start  ", self.XYZD_info.shape)
                    XYZD_info1 = read_pcap_files(pcap_dir, pacTimeStamps)
                    if XYZD_info1.empty == False:
                        print("return ", XYZD_info1.shape)

                        XYZD_info = pd.concat([self.XYZD_info, XYZD_info1], ignore_index=True)
                        print("concat ", XYZD_info.shape)

                        XYZD_info['counter'] = XYZD_info.duplicated(["azimuth", "laser_id"],
                                                                    keep="last")
                        XYZD_info = XYZD_info[XYZD_info['counter'] == False]
                        # XYZD_info = XYZD_info.drop(['counter', 'flag'], axis=1)
                        XYZD_info = XYZD_info.drop(['counter'], axis=1)
                        print("afret deleting duplicated ", XYZD_info.shape)

    def camera_timestamps_processing(self, shot, key, value, image_number):
        # print ("camera ")
        image_number += 1
        leftImages = shot['leftImage']
        YAML_Image_names = ("new." + str(self.VideoFlows[-1]) + '.' + str(
            self.VideoNumbers[-1]) + '.' + 'left.' + str('%000006d' % leftImage_FrameNumber))
        # print ("image ")
        for key, value in leftImages.items():
            if key.startswith("deviceSec"):
                leftImage_deviceSec = int(key[len("deviceSec:"):])
            if key.startswith("grabMsec"):
                leftImage_grabMsec = int(key[len("grabMsec:"):])

                # print ("leftImage_grabMsec ",leftImage_grabMsec)
        return 0

    @staticmethod
    def give_real_val(dis, azimuth, laser_id):
        r = dis * DISTANCE_RESOLUTION
        omega = LASER_ANGLES[laser_id] * np.pi / 180.0
        alpha = azimuth / 100.0 * np.pi / 180.0
        x = r * np.cos(omega) * np.sin(alpha)
        y = r * np.cos(omega) * np.cos(alpha)
        z = r * np.sin(omega)
        return x, y, z


def read_pcap_files(pcap_dir, pacTimeStamps):
    files = natsorted(glob.glob(pcap_dir + '/*.pcap'))
    data1 = {'X': [], 'Y': [], 'Z': [], 'ref': [], 'azimuth': [], 'laser_id': []}
    df1 = pd.DataFrame(data1)
    for x in files:
        d = open(x, 'rb').read()
        n = len(d)
        packet = d[24:]  # packet header and packet data  without global header
        for offset in range(0, n - 24, 1264):
            if (n - offset) < 1264: break
            data = packet[offset + 16 + 42: offset + 16 + 42 + 1200 + 4 + 2]
            first_timestamp, factory = struct.unpack_from("<IH", data,
                                                          offset=1200)  # timestamp for the first firing in the packet data
            # print ("first_timestamp ",first_timestamp)
            # print ("pacTimeStamps   ",pacTimeStamps[-1])
            # print ("  ")
            assert hex(factory) == '0x2237', 'Error mode: 0x22=VLP-16, 0x37=Strongest Return'
            if (first_timestamp * 1.0) in pacTimeStamps:
                # print ("pcap  ", x )
                seq_index = 0
                for seq_offset in range(0, 1200, 100):
                    flag, first_azimuth = struct.unpack_from("<HH", data, seq_offset)
                    assert hex(flag) == '0xeeff', 'Flag error'
                    for step in range(2):
                        if (step == 0) and ((seq_index % 2) == 0) and (seq_index < 22):
                            flag, third_azimuth = struct.unpack_from("<HH", data, seq_offset + 4 + 3 * 16 * 2)
                            assert hex(flag) == '0xeeff', 'Flag error'
                            if (third_azimuth < first_azimuth):
                                step_azimuth = third_azimuth + ROTATION_MAX_UNITS - first_azimuth
                            else:
                                step_azimuth = third_azimuth - first_azimuth
                        arr = struct.unpack_from('<' + "HB" * NUM_LASERS, data, seq_offset + 4 + step * 3 * 16)
                        for i in range(NUM_LASERS):
                            azimuth = first_azimuth + (step_azimuth * (55.296 / 1e6 * step + i * (2.304 / 1e6))) / (
                                        2 * 55.296 / 1e6)
                            if (azimuth > ROTATION_MAX_UNITS):
                                azimuth -= ROTATION_MAX_UNITS
                            if arr[i * 2] != 0:
                                X, Y, Z = calc_real_val(arr[i * 2], azimuth, i)
                                # azimuth_time = (55.296/1e6 * step +i * (2.304/1e6))+first_timestamp
                                if Y > 0:
                                    new_row = pd.Series([X, Y, Z, arr[i * 2 + 1], round(azimuth * 1.0 / 100), i],
                                                        index=df1.columns)
                                    # new_row = pd.Series(data={'X': X, 'Y': Y, 'Z': Z, 'ref': arr[i * 2 + 1], 'azimuth': int(azimuth/1000),'laser_id': i}, name=round(azimuth_time*1.0))
                                    # print ("new ", new_row)
                                    df1 = df1.append(new_row, ignore_index=True)
                                    # df = df.drop_duplicates(subset=['azimuth', 'laser_id'], keep='last')
                                    # print ("after ", df [1020:1081])
                        seq_index += 1
            else:
                # print(first_timestamp)#, " ", pacTimeStamps)
                break

    # print ("after ", df1)# [1020:1081])

    # df2 = df.drop_duplicates(subset=['azimuth', 'laser_id'], keep='last')
    # df=df2
    x = None
    return (df1)
