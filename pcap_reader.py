import struct
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

import time

import progressbar #pip install progressbar2

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

class PointReader:
    def __init__(self, data_path):
        self.data_path = Path(data_path) / 'data'
        self.pcap_path = self.data_path / 'pcap'
        self.pcap_files = [x for x in self.pcap_path.glob('*.*') if x.is_file()]
        self.LASER_ANGLES = [-15, 1, -13, -3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        self.NUM_LASERS = 16
        self.DISTANCE_RESOLUTION = 0.002
        self.ROTATION_MAX_UNITS = 36000

        self.df = pd.DataFrame({'X': [], 'Y': [], 'Z': [], 'D': [], 'azimuth': [], 'laser_id': [],
                                'first_timestamp': [], 'pcap_num': []})
        self.local_df = []

        self.azimuth_bin = 100
        self.first_timestamp = None
        self.factory = None

    def read_pcap(self, file_number):
        with Timer() as t:
            self.get_pcap_data(file_number)
        print('PointReader.read_pcap finished in %.03f s' % t.interval)

        return self.df

    def get_pcap_data(self, file_number):

        pcap_file = str(self.pcap_files[int(file_number) - 1])
        pcap_data = open(pcap_file, 'rb').read()
        pcap_data = pcap_data[24:]

        with progressbar.ProgressBar(max_value=int(len(pcap_data))) as bar:
            for offset in range(0, int(len(pcap_data)), 1264):
                counter = offset // 1264
                bar.update(offset)
                # if counter % 30 == 0:
                #    print("offset=", offset, int(len(pcap_data)))
                    # if counter > 1:
                    #     return

                if (len(pcap_data) - offset) < 1264:
                    break

                cur_packet = pcap_data[offset + 16: offset + 16 + 42 + 1200 + 4 + 2]
                cur_data = cur_packet[42:]

                self.first_timestamp, self.factory = struct.unpack_from("<IH", cur_data, offset=1200)
                assert hex(self.factory) == '0x2237', 'Error mode: 0x22=VLP-16, 0x37=Strongest Return'
                seq_index = 0

                for seq_offset in range(0, 1100, 100):
                    self.seq_processing(cur_data, seq_offset, seq_index, self.first_timestamp, int(file_number) - 1)

        self.df = pd.DataFrame(self.local_df)

        #print('End processing pcap file...')

    def seq_processing(self, data, seq_offset, seq_index, first_timestamp, pcap_num):
        flag, first_azimuth = struct.unpack_from("<HH", data, seq_offset)
        step_azimuth = 0

        assert hex(flag) == '0xeeff', 'Flag error'

        data_dict_list = []

        for step in range(2):
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
                if y > 0:
                # azimuth_time = (55.296 / 1e6 * step + i * (2.304 / 1e6)) + first_timestamp
                    d = arr[i * 2 + 1]
                    azimuth_v = round(azimuth * 1.0 / self.azimuth_bin)
                    data_dict_list.append({'X': x, 'Y': y, 'Z':z, 'D': d, 'azimuth': azimuth_v, 'laser_id': i, 
                        'first_timestamp': first_timestamp, 'pcap_num': pcap_num})
                    # new_row = pd.Series(
                    #         [x, y, z, arr[i * 2 + 1], round(azimuth * 1.0 / self.azimuth_bin), i, first_timestamp, pcap_num],
                    #         index=self.df.columns)
                    # self.local_df = self.local_df.append(new_row, ignore_index=True)
            seq_index += 1
        self.local_df += data_dict_list

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
        self.processed_pcap = None
        self.full_dataframe = pd.DataFrame()

    def get_all_points_by_timestamp(self, timestamp, pcap_index, read_next_pcap_file, azimuth=None):

        if pcap_index != self.processed_pcap:
            self.processed_pcap = pcap_index
            index_dataframe = self.local_point_reader.read_pcap(pcap_index)
            self.full_dataframe = pd.concat([self.full_dataframe, index_dataframe], ignore_index=True)
            # print(timestamp, self.full_dataframe['first_timestamp'])

        if read_next_pcap_file is not False:
            print("self.processed_pcap ",self.processed_pcap)
            print('pcap_index',pcap_index)
            pcap_number= int(pcap_index)+1
            self.processed_pcap = "00"+str(pcap_number)
            print("self.processed_pcap ", self.processed_pcap)
            index_dataframe = self.local_point_reader.read_pcap(self.processed_pcap)
            self.full_dataframe = pd.concat([self.full_dataframe, index_dataframe], ignore_index=True)
            print(timestamp, self.full_dataframe['first_timestamp'])

        if azimuth is not None:
            return self.full_dataframe[(self.full_dataframe['azimuth'] == azimuth) &
                                        [self.full_dataframe['first_timestamp'] == timestamp]]
        else:
            #print(self.full_dataframe)
            return self.full_dataframe[self.full_dataframe['first_timestamp'] == timestamp]