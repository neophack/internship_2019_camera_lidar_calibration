import re
import gzip
from datetime import datetime
import yaml
from pathlib import Path
import pandas as pd
import numpy as np


from pcap_reader import PointProcessing
from logger import LogWriter


class YML_Read:
    def __init__(self, data_path):

        self.path = Path(data_path) / 'data'
        self.pcap_path = self.path / 'pcap'
        self.yml_path = self.path / 'yml'
        self.images = self.path / 'frames'

        self.pcap_files = np.sort([x for x in self.pcap_path.glob('*.pcap') if x.is_file()])
        self.yml_files = np.sort([x for x in self.yml_path.glob('*.gz*') if x.is_file()])
        # cross-platform way to get file name without extention
        self.image_files = [x.stem for x in self.images.glob('*.bmp') if x.is_file()]

        self.processing = PointProcessing(data_path)
        self.writer = LogWriter(data_path)

        self.processed_images = []
        self.VideoFlows = []
        self.VideoNumbers = []
        self.time_lidar = None
        self.image_number = 0
        self.XYZD_info = pd.DataFrame({'X': [], 'Y': [], 'Z': [], 'D': [], 'azimuth': [], 'laser_id': [],
                                       'first_timestamp': [], 'pcap_num': []})

    def power(self, range_yml=None):
        print('Running main loop...')
        if range_yml is None:
            range_yml = np.arange(len(self.yml_files))
        for i in range_yml:
            self.image_number = 0
            yml_file = self.yml_files[i]
            file_name = yml_file.stem # cross-platform way to get file name without extention

            self.regular_expression(yml_file=file_name)
            print(f'Reading {file_name} ...')
            self.read_yml(filename=yml_file)

    def regular_expression(self, yml_file):
        mat = re.match(r"(?P<flow>\S+)\.(?P<VideoFlow>\d+)\.(?P<VideoNumber>\d+)\.(?P<info>\S+)\.(?P<type>\d*)",
                       yml_file)
        mat = mat.groupdict()
        self.VideoNumbers.append(mat["VideoNumber"])
        self.VideoFlows.append(mat["VideoFlow"])

    def read_yml(self, filename):
        with gzip.open(filename, "rt") as file:
            config = file.read()
            self.image_number = 0
            if config.startswith("%YAML:1.0"):
                config = "%YAML 1.1" + str(config[len("%YAML:1.0"):])
                data = list(yaml.safe_load_all(config))

                for shot in (data[0]['shots']):
                    self.shot_processing(shot=shot)

    def shot_processing(self, shot):
        for key, value in sorted(shot.items(), reverse=True):
            if key.startswith("velodyneLidar"):
                pacTimeStamps = shot['velodyneLidar']["lidarData"]["pacTimeStamps"]
                self.lidar_timestamps_processing(pacTimeStamps[-1])

            if key.startswith("leftImage"):
                yaml_img_name, leftImage_deviceSec, \
                leftImage_grabMsec = self.camera_timestamps_processing(shot['leftImage'].items())

                if yaml_img_name in self.image_files and yaml_img_name not in self.processed_images:

                    self.processed_images.append(yaml_img_name)
                    self.writer.save_images(yaml_img_name=yaml_img_name,
                                            image_time=(leftImage_grabMsec / 1e6 + leftImage_deviceSec))
                    time_lidar = datetime.fromtimestamp(
                        leftImage_grabMsec / 1e6 + leftImage_deviceSec).strftime('%Y-%m-%d_%H_%M_%S.%f')
                    self.writer.save_lidar_data(time_lidar=time_lidar, df=self.XYZD_info)

    def lidar_timestamps_processing(self, last_pacTimeStamp):
        import datetime
        start = datetime.datetime.now()
        XYZD_info_temp = self.processing.get_all_points_by_timestamp(last_pacTimeStamp,
                                                                     pcap_index=self.VideoNumbers[-1],
                                                                     read_next_pcap_file = False)
        if not XYZD_info_temp.empty:

            #print("XYZD_info_temp", XYZD_info_temp)

            self.XYZD_info = pd.concat([self.XYZD_info, XYZD_info_temp], ignore_index=True)
            self.XYZD_info['counter'] = self.XYZD_info.duplicated(["azimuth", "laser_id"], keep="last")
            self.XYZD_info = self.XYZD_info[self.XYZD_info['counter'] == False]
            self.XYZD_info = self.XYZD_info.drop(['counter'], axis=1)
        dt = datetime.datetime.now() - start
        #print('lidar_timestamps_processing finished in', dt, ' sec')

        #print("XYZD_info ", self.XYZD_info)

    def camera_timestamps_processing(self, camera_items):
        leftImage_deviceSec = None
        leftImage_grabMsec = None

        yaml_img_name = ("new." + str(self.VideoFlows[-1]) + '.' + str(
            self.VideoNumbers[-1]) + '.' + 'left.' + str('%000006d' % self.image_number))
        self.image_number += 1

        for key_Image, value_Image in camera_items:
            if key_Image.startswith("deviceSec"):
                leftImage_deviceSec = int(key_Image[len("deviceSec:"):])
            if key_Image.startswith("grabMsec"):
                leftImage_grabMsec = int(key_Image[len("grabMsec:"):])
        return yaml_img_name, leftImage_deviceSec, leftImage_grabMsec