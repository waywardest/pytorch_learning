from __future__ import print_function
import six
import os  # needed navigate the system to get the input data
import numpy as np
import radiomics
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import argparse
import gzip
def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    #获取文件的名称，去掉
    g_file = gzip.GzipFile(file_name)
    #创建gzip对象
    open(f_name, "wb+").write(g_file.read())
    #gzip对象用read()打开后，写入open()建立的文件里。
    g_file.close() #关闭gzip对象

def catch_features(imagePath,maskPath):
    if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
        raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
    settings = {}
    settings['binWidth'] = 25  # 5
    settings['sigma'] = [3, 5]
    settings['Interpolator'] = sitk.sitkBSpline
    settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
    settings['voxelArrayShift'] = 1000  # 300
    settings['normalize'] = True
    settings['normalizeScale'] = 100
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    #extractor = featureextractor.RadiomicsFeatureExtractor()
    print('Extraction parameters:\n\t', extractor.settings)

    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation', 'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
    extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion','Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn','Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness'])
# 上边两句我将一阶特征和形状特征中的默认禁用的特征都手动启用，为了之后特征筛选
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    feature_cur = []
    feature_name = []
    result = extractor.execute(imagePath, maskPath, label=1)
    for key, value in six.iteritems(result):
        print('\t', key, ':', value)
        feature_name.append(key)
        feature_cur.append(value)
    print(len(feature_cur[37:]))
    name = feature_name[37:]
    name = np.array(name)
    '''
    flag=1
    if flag:
        name = np.array(feature_name)
        name_df = pd.DataFrame(name)
        writer = pd.ExcelWriter('key.xlsx')
        name_df.to_excel(writer)
        writer.save()
        flag = 0
    '''
    for i in range(len(feature_cur[37:])):
        #if type(feature_cur[i+22]) != type(feature_cur[30]):
        feature_cur[i+37] = float(feature_cur[i+37])
    return feature_cur[37:],name

image_dir = r'Z:\alwang\feature_extract'
mask_dir = r'Z:\alwang\feature_extract'
patient_list = os.listdir(image_dir)
#print(patient_list)
save_file = np.empty(shape=[1,1051])
id = []
for patient in patient_list:
    print(patient)
    if patient =='DKI_DKI_AD.nii':
        imagePath = os.path.join(image_dir,patient)
    if patient =='voi.nii':
        maskPath = os.path.join(mask_dir,patient)
print(imagePath)
print(maskPath)
save_curdata,name = catch_features(imagePath,maskPath)
save_curdata = np.array(save_curdata)
save_curdata = save_curdata.reshape([1, 1051])
print(save_curdata)
#for patient in patient_list:
    #id.append(patient.split('.')[0])
    #np.concatenate((patient,save_curdata),axis=1)
save_file = np.append(save_file,save_curdata,axis=0)
print(save_file.shape)
save_file = np.delete(save_file,0,0)
#save_file = save_file.transpose()
#print(save_file.shape)
id=['1']
id_num = len(id)
id = np.array(id)
name_df = pd.DataFrame(save_file)
name_df.index = id
name_df.columns = name
writer = pd.ExcelWriter('NSCLC-Radiomics-features.xlsx')
name_df.to_excel(writer)
writer.save()
