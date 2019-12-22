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


class Engine:
    '''main tread - reading yml files from "data/yml" dir'''
    def __init__(self, data_path):
        self.path = Path(data_path) / 'data'
        self.pcap_path = self.path / 'pcap'
        self.yml_path = self.path / 'yml'
        self.images = self.path / 'frames'
        self.pcap_files = [x for x in self.pcap_path.glob('*.*') if x.is_file()]
        self.yml_files = [x for x in self.yml_path.glob('*.*') if x.is_file()]
        self.image_files = [x for x in self.images.glob('*.bmp') if x.is_file()]

        self.processing = PointProcessing(data_path)
        self.writer = LogWriter(data_path)
        self.processed_images = []
        self.VideoFlows = []
        self.VideoNumbers = []
        self.time_lidar = None
        #self.leftImage_grabMsec = None
        #self.leftImage_deviceSec = None
        self.pacTimeStamps = None
        self.XYZD_info = pd.DataFrame({'X': [], 'Y': [], 'Z': [], 'D': [], 'azimuth': [], 'laser_id': [], 'first_timestamp': [],
                                      'pcap_num': []})

    def power(self, step):
        '''read yml file №step from yml files'''
        yml_file = str(self.yml_files[step])
        file_name = (yml_file.split('\\')[-1])[:-3]
        print(file_name)
        self.regular_expression(yml_file=file_name)
        self.read_yml(filename=yml_file)

    def regular_expression(self, yml_file):
        '''parsing yml file name'''
        mat = re.match(r"(?P<flow>\S+)\.(?P<VideoFlow>\d+)\.(?P<VideoNumber>\d+)\.(?P<info>\S+)\.(?P<type>\d*)",
                       yml_file)
        mat = mat.groupdict()
        self.VideoNumbers.append(mat["VideoNumber"])
        self.VideoFlows.append(mat["VideoFlow"])

    def read_yml(self, filename):
        with gzip.open(filename, "rt") as file:
            config = file.read()
            image_number = 0
            if config.startswith("%YAML:1.0"):
                config = "%YAML 1.1" + str(config[len("%YAML:1.0"):])
                data = list(yaml.safe_load_all(config))
                for shot in (data[0]['shots']):
                    self.shot_processing(image_number=image_number, shot=shot)

    def shot_processing(self,  image_number, shot):
        for key, value in sorted(shot.items(), reverse=True):
            if key.startswith("velodyneLidar"):
                pacTimeStamps = shot['velodyneLidar']["lidarData"]["pacTimeStamps"]
                self.lidar_timestamps_processing(pacTimeStamps)

            if key.startswith("leftImage"):
                print("LeftImage")
                yaml_img_name, leftImage_deviceSec, leftImage_grabMsec  = self.camera_timestamps_processing(shot, image_number)
                """for key_Image, value_Image in shot['leftImage'].items():
                    leftImage_deviceSec = int(key_Image[len("deviceSec:"):])
                    print(leftImage_deviceSec)
                    leftImage_grabMsec = int(key_Image[len("grabMsec:"):])
                    print(leftImage_grabMsec)"""

                if yaml_img_name in [str(x) for x in self.image_files] and yaml_img_name not in self.processed_images:

                    self.processed_images.append(yaml_img_name)
                    self.writer.save_images(yaml_img_name=yaml_img_name,
                                            image_time=(leftImage_grabMsec / 1e6 + leftImage_deviceSec))
                    time_lidar = datetime.fromtimestamp(
                        self.leftImage_grabMsec / 1e6 + self.leftImage_deviceSec).strftime('%Y-%m-%d_%H_%M_%S.%f')
                    self.writer.save_lidar_data(time_lidar=time_lidar, df=self.XYZD_info)

    def lidar_timestamps_processing(self, pacTimeStamps):
        for pacTimeStamp in pacTimeStamps:
            print(pacTimeStamp)
            XYZD_info_temp = self.processing.get_points_by_timestamp(pacTimeStamp, self.VideoNumbers[-1])
            if not XYZD_info_temp.empty:
                print('new data_frame ', XYZD_info_temp.shape)
                print('old data_frames ', self.XYZD_info.shape)
                self.XYZD_info = pd.concat([self.XYZD_info, XYZD_info_temp], ignore_index=True)
                print('after concat ', self.XYZD_info.shape)
                self.XYZD_info['counter'] = self.XYZD_info.duplicated(["azimuth", "laser_id"],
                                                            keep="last")
                self.XYZD_info = self.XYZD_info[self.XYZD_info['counter'] == False]
                self.XYZD_info = self.XYZD_info.drop(['counter'], axis=1)
                print("after deleting duplicated ", self.XYZD_info.shape)

    def camera_timestamps_processing(self, shot, image_number):
        image_number += 1
        yaml_img_name = ("new." + str(self.VideoFlows[-1]) + '.' + str(
            self.VideoNumbers[-1]) + '.' + 'left.' + str('%000006d' % image_number))
        leftImage_deviceSec = int(shot['leftImage']["deviceSec"][len("deviceSec:"):])
        leftImage_grabMsec = int(shot['leftImage']["grabMsec"][len("grabMsec:"):])
        return yaml_img_name, leftImage_deviceSec, leftImage_grabMsec


class PointReader:
    def __init__(self, data_path):
        self.data_path = Path(data_path) / 'data'
        self.pcap_path = self.data_path / 'pcap'
        self.pcap_files = [x for x in self.pcap_path.glob('*.*') if x.is_file()]
        self.LASER_ANGLES = [-15, 1, -13, -3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        self.NUM_LASERS = 16
        self.DISTANCE_RESOLUTION = 0.002
        self.ROTATION_MAX_UNITS = 36000

        self.df = pd.DataFrame({'X': [], 'Y': [], 'Z': [], 'D': [], 'azimuth': [], 'laser_id': [], 'first_timestamp': [],
                                'pcap_num': []})
        #self.all_packets = bytes(0)
        self.azimuth_bin = 1000
        self.first_timestamp = None
        self.factory = None

    def read_pcap(self, file_number):
        self.get_pcap_data(file_number)
        return self.df

    def get_pcap_data(self, file_number):
        pcap_file = str(self.pcap_files[int(file_number)-1])
        pcap_data = open(pcap_file, 'rb').read()
        pcap_data = pcap_data[24:]  #global header
        for offset in range(0, len(pcap_data), 1264):
            if (len(pcap_data) - offset) < 1264 : break
            print(offset , len(pcap_data))
            cur_packet= pcap_data[offset + 16 : offset + 16 + 42 + 1200 + 4 + 2]  #current packet 1264 inclide 16 timeinfo
            cur_data = cur_packet[42 :]
            #self.all_packets += cur_data #1206
            self.first_timestamp, self.factory = struct.unpack_from("<IH", cur_data, offset =  1200)
            assert hex(self.factory) == '0x2237', 'Error mode: 0x22=VLP-16, 0x37=Strongest Return'
            seq_index = 0
            for seq_offset in range(0, 1100, 100):
                self.seq_processing(cur_data, seq_offset, seq_index, self.first_timestamp, file_number)

    def seq_processing(self, data, seq_offset, seq_index, first_timestamp, pcap_num):
        flag, first_azimuth = struct.unpack_from("<HH", data, seq_offset)
        step_azimuth = 0
        assert hex(flag) == '0xeeff', 'Flag error'
        for step in range(2):
            #print (step, seq_index, seq_offset)
            if step == 0 and seq_index % 2 == 0 and seq_index < 22:
                flag, third_azimuth = struct.unpack_from("<HH", data, seq_offset + 4 + 3 * 16 * 2)
                assert hex(flag) == '0xeeff', 'Flag error'
                if third_azimuth < first_azimuth:
                    step_azimuth = third_azimuth + self.ROTATION_MAX_UNITS - first_azimuth
                else:
                    step_azimuth = third_azimuth - first_azimuth

            arr = struct.unpack_from('<' + "HB" * self.NUM_LASERS, data, seq_offset + 4 + step * 3 * 16)

            for i in range(self.NUM_LASERS):
                azimuth = first_azimuth + (step_azimuth * (55.296 / 1e6 * step + i * 2.304 / 1e6)) / (2 * 55.296 / 1e6)
                if azimuth > self.ROTATION_MAX_UNITS:
                    azimuth -= self.ROTATION_MAX_UNITS

                x, y, z = self.calc_real_val(arr[i * 2], azimuth, i)
                # azimuth_time = (55.296 / 1e6 * step + i * (2.304 / 1e6)) + first_timestamp
                new_row = pd.Series([x, y, z, arr[i * 2 + 1], round(azimuth * 1.0 / self.azimuth_bin), i, first_timestamp, pcap_num],
                                    index=self.df.columns)
                self.df = self.df.append(new_row, ignore_index=True)
            seq_index += 1

    def calc_real_val(self, dis, azimuth, laser_id):
        r = dis * self.DISTANCE_RESOLUTION
        omega = self.LASER_ANGLES[laser_id] * np.pi / 180.0
        alpha = (azimuth / 100.0) * (np.pi / 180.0)
        x = r * np.cos(omega) * np.sin(alpha)
        y = r * np.cos(omega) * np.cos(alpha)
        z = r * np.sin(omega)

        return x, y, z


class PointProcessing:
    def __init__(self, data_path):
        self.data_path = data_path
        self.local_point_reader = PointReader(data_path=self.data_path)

    def refresh_reader(self):
        self.local_point_reader = PointReader(data_path=self.data_path)

    def get_points_by_timestamp(self, timestamp, VideoNumber):
        dataframe = self.local_point_reader.read_pcap(VideoNumber)
        return dataframe[dataframe['first_timestamp'] == timestamp]


class LogWriter:
    def __init__(self, data_path):

        self.data_path = data_path
        self.results_path = Path(data_path) / 'results'
        if self.results_path.is_dir():
            shutil.rmtree(str(self.results_path), ignore_errors=True)
        self.results_path.mkdir()

        self.calib_path= self.results_path / 'calib'
        self.image_path = self.results_path / 'leftImage'
        self.lidar_path = self.results_path / 'velodyne_points'
        self.image_data_path = self.image_path / 'data'
        self.lidar_data_path = self.lidar_path / 'data'

        self.calib_path.mkdir()
        self.image_path.mkdir()
        self.lidar_path.mkdir()
        self.image_data_path.mkdir()
        self.lidar_data_path.mkdir()

        #copy_tree(str(Path(self.data_path) / 'frames' / 'calib'), str(self.calib_path))
        open(str(self.image_path / 'timestamps.txt'), "w+")
        open(str(self.lidar_path / 'timestamps.txt'), "w+")

        self.indx = 0

    def save_images(self, yaml_img_name, image_time):
        image_files = [str(x) for x in (Path(self.data_path) / 'data' / 'frames').glob('*.bmp') if x.is_file()]

        for item in image_files:
            if item.split('\\')[-1][:-3] == yaml_img_name:
                shutil.copy(item, str(self.image_data_path))

        with open(str(self.image_path / 'timestamps.txt'), "a+") as f:
            f.write("%s\n" % datetime.fromtimestamp(image_time).strftime('%Y-%m-%d %H:%M:%S.%f'))

    def save_lidar_data(self, time_lidar, df):
        self.indx += 1
        with open(str(self.lidar_path / 'timestamps.txt'), "a+") as f:
            f.write("%s\n" % time_lidar)

        path = str(self.lidar_data_path / (str(self.indx) + '.txt'))
        self.save_dataframe(df, path=path)

    def save_dataframe(self, df, path=None):
        if path is not None:
            df.to_csv(path, header=False, index=False)
        else:
            results_path = Path(self.data_path) / 'results'
            if results_path.is_dir():
                shutil.rmtree(str(results_path))
            results_path.mkdir()
            df.to_csv(str(results_path / 'main_dataframe.csv'))