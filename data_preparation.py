#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
import os, random, shutil
import pandas as pd


# Read file with the mapping between the images and the objid DR7
df_mapping = pd.read_csv('/home/pcunha/Documents/Summer_interships/gz2_filename_mapping.csv')
#df_class = pd.read_csv('zoo2MainSpecz.csv')

# Read file with the GZ2 classifications
df_class = pd.read_csv('/home/pcunha/Documents/Summer_interships/gz2_hart16.csv')

# Renames and merge DataFrames using the objid as reference
df_class.rename(columns = {'dr7objid':'objid'}, inplace = True)

df_combine = df_class.merge(df_mapping, how='left', on='objid')

# Creates a binary class based on the first letter of the GZ2 classification
df_combine['g_class'] = df_combine['gz2_class'].astype(str).str[0]

# Path where I have the images. Change this accordingly!
base_path = r'/home/pcunha/Documents/Summer_interships/images_gz2'

#Creates the path where the images are stored
training_images = os.path.join(base_path, 'images')

# Sanity check to prevent possible errors. Checks if the asset_id files exists in the image folder.
image_files = pd.Series(os.listdir(training_images), name='asset_id')
image_files = image_files.str.split('.').str[0].astype(int)

df_combine['image_exist'] = df_combine['asset_id'].isin(image_files)
df_id_images = df_combine.copy()
# Only sources with images in the folder will be used hereafter
df_id_images = df_id_images[df_id_images['image_exist'] == True]

# Extract a number of images to be used for classification
n_source = 2000
ellipticals = df_id_images['asset_id'][df_id_images['g_class']=='E'][:n_source].to_list()
spirals = df_id_images['asset_id'][df_id_images['g_class']=='S'][:n_source].to_list()

#Path that will be used to store the folder and copy of the images
my_data = os.path.join(base_path, 'data')

def _proc_images(src, dst, label, arr, percent):
    '''
    src: path for the images folder
    dst: path to be created
    label: label of the galaxy
    arr: array with the IDs for a given class
    percent: percentage to be used for training. Should be between 0.1 and 0.9 (recommended 0.7)
    '''

    train_dir = os.path.join(dst, 'train')
    val_dir = os.path.join(dst, 'validation')
    
    train_dest = os.path.join(train_dir, label)
    val_dest   = os.path.join(val_dir, label)
    
    if not os.path.exists(train_dest):
        os.makedirs(train_dest)

    if not os.path.exists(val_dest):
        os.makedirs(val_dest)
    
    random.shuffle(arr)
    
    idx = int(len(arr)*percent)
    for i in arr[0:idx]:
        shutil.copyfile(os.path.join(src, str(i)+'.jpg'), os.path.join(train_dest, str(i)+'.jpg'))
    for i in arr[idx:]:
        shutil.copyfile(os.path.join(src, str(i)+'.jpg'), os.path.join(val_dest, str(i)+'.jpg'))
    
    print(label, 'done!')


#Run the previous function for each class of interest: elliptical and spiral. Using 70% of the images for training, and 30% for validation.
_proc_images(training_images, my_data, 'elliptical', ellipticals, 0.7)
_proc_images(training_images, my_data, 'spiral', spirals, 0.7)
