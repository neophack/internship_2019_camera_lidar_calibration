from __future__ import absolute_import
import os,re
import sys
import glob
import pandas as pd
import struct
from natsort import natsorted
import numpy as np
import os.path
import gzip
from datetime import datetime
from distutils.dir_util import copy_tree
import yaml
import shutil
from distutils import dir_util
import calendar

"""
Takes raw PCAP folder and photo_folder with timestamp  from YAML (result of script 1) 
Add XYZD PCAP data to photo_folder according YAML data into "velodyne_points" subfolder 
"""

LASER_ANGLES = [-15, 1, -13, -3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
NUM_LASERS = 16
DISTANCE_RESOLUTION = 0.002
ROTATION_MAX_UNITS = 36000

def print_help_and_exit1():
    print('Usage:YAML dir, frames dir, PCAP dir')
    sys.exit()

def calc_real_val(dis, azimuth, laser_id):
    R = dis * DISTANCE_RESOLUTION
    omega = LASER_ANGLES[laser_id] * np.pi / 180.0
    alpha = azimuth / 100.0 * np.pi / 180.0
    X = R * np.cos(omega) * np.sin(alpha)
    Y = R * np.cos(omega) * np.cos(alpha)
    Z = R * np.sin(omega)
    return (X, Y, Z)

def delete_files_in_dir(path):
    files = glob.glob(path+'/*.')
    for f in files:
        os.remove(f)
    
def get_image_names(frames_dir):
    """get the list of image names in images file folder"""
    Image_names = []
    for f in sorted(os.listdir(frames_dir)):
        if os.path.splitext(f)[-1] not in {'.bmp', '.png', '.jpg'}:
            continue
        f_names = os.path.split(os.path.splitext(f)[0])[-1] 
        Image_names.append(f_names)
    return Image_names

def move_images(counter, frames_dir, YAML_Image_name, out, image_time):
    dev_path = os.path.join(out, 'leftImage')
    data_path = os.path.join(dev_path, 'data')
    
    if counter == 0:
        if os.path.exists(dev_path) is False:
            os.makedirs(dev_path)
            os.makedirs(data_path)
            open(dev_path+'/timestamps.txt',"w+")
            os.makedirs(os.path.join(out, 'calib'))
        else: 
            delete_files_in_dir(dev_path)
            delete_files_in_dir(data_path)
      
    copy_tree(frames_dir+"/calib", out+"/calib")
    for item in glob.iglob(os.path.join(frames_dir, "*.bmp")):
        if (os.path.split(os.path.splitext(item)[0])[-1]) == YAML_Image_name:
            shutil.copy(item, data_path)
    f = open(dev_path+'/timestamps.txt', "a+")
    f.write("%s\n" % datetime.fromtimestamp(image_time).strftime('%Y-%m-%d %H:%M:%S.%f'))
    f.close() 

def save_lidar_data_time(time_lidar, out, counter, XYZD_info):
    velodyne_time_path = os.path.join(out, 'velodyne_points') 
    velodyne_data_path = os.path.join(velodyne_time_path, 'data/') 
        
    if counter == 0:
        if os.path.exists(velodyne_time_path) is False:
            os.makedirs(velodyne_time_path)  
            os.makedirs(velodyne_data_path) 
        else: 
            delete_files_in_dir(velodyne_time_path)  
            delete_files_in_dir(velodyne_data_path)  

    f = open(velodyne_time_path+'/timestamps.txt', "a+")
    f.write("%s\n" % time_lidar)
    f.close()        
            
    f1 = open(velodyne_data_path+ str(counter)+'.txt',"a+")
    f1.write(XYZD_info.to_string(header = False, index = False))
    f1.close()  
        
def read_pcap_files(pcap_dir, pacTimeStamps,df):
    files = natsorted(glob.glob(pcap_dir+'/*.pcap'))
    for x in files:
        d = open(x, 'rb').read()  
        n = len(d)
        packet = d[24 : ]        #packet header and packet data  without global header
        for offset in range(0, n-24, 1264):
            if (n-offset) < 1264: break  
            data = packet[offset + 16 + 42 : offset + 16 + 42 + 1200 + 4 + 2]
            first_timestamp, factory = struct.unpack_from("<IH", data, offset=1200)  #timestamp for the first firing in the packet data
            assert hex(factory) == '0x2237', 'Error mode: 0x22=VLP-16, 0x37=Strongest Return'
            if (first_timestamp in pacTimeStamps):
                seq_index = 0  
                for seq_offset in range(0, 1200, 100):
                    flag, first_azimuth = struct.unpack_from("<HH", data, seq_offset)
                    assert hex(flag) == '0xeeff' , 'Flag error'
                    for step in range(2):
                        if (step==0) and ((seq_index%2)==0) and (seq_index<22):
                            flag, third_azimuth = struct.unpack_from("<HH", data, seq_offset + 4 + 3 * 16 * 2  )
                            assert hex(flag) == '0xeeff' , 'Flag error'
                            if (third_azimuth<first_azimuth): 
                                step_azimuth=third_azimuth+ROTATION_MAX_UNITS-first_azimuth
                            else:
                                step_azimuth=third_azimuth-first_azimuth
                        arr = struct.unpack_from('<' + "HB" * NUM_LASERS, data, seq_offset + 4 + step * 3 * 16) 
                        for i in range(NUM_LASERS): 
                            azimuth = first_azimuth+ (step_azimuth * (55.296/1e6 * step +i * (2.304/1e6)))/(2 * 55.296/1e6)
                            if (azimuth>ROTATION_MAX_UNITS): 
                                azimuth-=ROTATION_MAX_UNITS       
                            if arr[i * 2] != 0:
                                X, Y, Z = calc_real_val(arr[i * 2], azimuth, i)
                                #azimuth_time = (55.296/1e6 * step +i * (2.304/1e6))+first_timestamp
                                if Y>0: 
                                    
                                    new_row =pd.Series([X, Y, Z, arr[i * 2 + 1],int(azimuth/100), i], index=df.columns )
                                    #new_row = pd.Series(data={'X': X, 'Y': Y, 'Z': Z, 'ref': arr[i * 2 + 1], 'azimuth': int(azimuth/1000),'laser_id': i}, name=int(azimuth_time))
                                    df = df.append(new_row, ignore_index=True)
                                    df = df.drop_duplicates(subset=['azimuth', 'laser_id'], keep='last')  
                        seq_index += 1 
            else:
                break
    return (df) 

def read_yaml(yaml_dir, frames_dir, pcap_dir):
    Image_names = get_image_names(frames_dir)
    Image_names_done = []
    YAML_Image_names = []
    VideoFlows = []
    VideoNumbers = []
    counter = 0
    files = natsorted(glob.glob(yaml_dir+ '\*.yml.gz'))
    time_lidar=None
    leftImage_grabMsec = None
    leftImage_deviceSec = None
    pacTimeStamps = None
    data = {'X': [],'Y': [],'Z': [],'ref': [],'azimuth': [],'laser_id':[]}
    XYZD_info= pd.DataFrame(data)
    
    for x in files:
        """read yaml file's name"""
        f_names = os.path.split(os.path.splitext(x)[0])[-1] 
        mat = re.match(r"(?P<flow>\S+)\.(?P<VideoFlow>\d+)\.(?P<VideoNumber>\d+)\.(?P<info>\S+)\.(?P<type>\d*)", f_names)
        for key,val in (mat.groupdict()).items():
            if key.startswith("VideoNumber"):
                VideoNumbers.append(val)
            if key.startswith("VideoFlow"):
                VideoFlows.append(val)   
                out =os.path.join("Kitti", VideoFlows[-1])
                
        """read yaml file's data"""        
        with gzip.open(x, "rt") as config:
            data = yaml.safe_load_all(config)
            c = config.read()
            leftImage_FrameNumber = 0
            if c.startswith("%YAML:1.0"):
                c = "%YAML 1.1" + str(c[len("%YAML:1.0"):]) 
                data = list(yaml.load_all(c))
                shots = data[0]['shots']
                for shot in shots:
                    for key, value in shot.items():
                        
                        """read camera timestamps and check is frame name in (good frames_dir)"""  
                        if  key.startswith("leftImage"):
                            leftImage_FrameNumber +=1    
                            leftImages=shot['leftImage']
                            #print ("image ")
                            for key, value in leftImages.items():
                                if  key.startswith("deviceSec"):
                                    leftImage_deviceSec=int(key[len("deviceSec:"):])
                                if  key.startswith("grabMsec"):
                                    leftImage_grabMsec=int(key[len("grabMsec:"):])
                                    
                            YAML_Image_names.append("new." + str(VideoFlows[-1]) + '.' + str(VideoNumbers[-1]) + '.' +
                                                    'left.'+str('%000006d' % leftImage_FrameNumber))
                            
                            
                        """read lidar timestamps data """        
                        if  key.startswith("velodyneLidar"):
                            velodyneLidars=shot['velodyneLidar']
                            for key, value in velodyneLidars.items():
                                if key.startswith("lidarData"):
                                    pacTimeStamps = value["pacTimeStamps"]
                                    XYZD_info = read_pcap_files(pcap_dir,pacTimeStamps, XYZD_info)
                                    print ("XYZD_info after  " ,XYZD_info.loc[20:50, ['X', 'azimuth', 'laser_id']] )
                              
                                
                    """move images from frame folder to Kitti struct with time from yaml_file"""          
                    if ((YAML_Image_names[-1] in Image_names) and (YAML_Image_names[-1] not in Image_names_done)):
                        Image_names_done.append(YAML_Image_names[-1])
                        move_images(counter, frames_dir, YAML_Image_names[-1], out,
                                            leftImage_grabMsec/1e6+leftImage_deviceSec)
                        time_lidar = datetime.fromtimestamp(leftImage_grabMsec/1e6+leftImage_deviceSec).strftime('%Y-%m-%d_%H_%M_%S.%f')
                        save_lidar_data_time(time_lidar, out, counter, XYZD_info)
                        counter+=1   
                    
def main():
    if len(sys.argv) == 4:
        print ('YAML dir:', sys.argv[1], ', frames dir:' , sys.argv[1],' PCAP dir:', sys.argv[3])
        read_yaml(sys.argv[1],sys.argv[2],sys.argv[3])   
    else:
        print_help_and_exit1()

if __name__ == '__main__':
    main()