import os
import glob
import cv2
import numpy as np
import pickle5 as pickle
import wget
from zipfile import ZipFile

# Read links from txt file
links_file_name = 'dataset_links.txt'


# Download all selected links to local drive
with open(links_file_name, 'r') as links_file:
	
	links = links_file.readlines()
	
	for link in links:
		print(link)
		# download file 
		ret = wget.download(link)

# unzipping all downloaded file
# Getting all filenames
zip_names = glob.glob('*.zip')

for filename in zip_names:
	with ZipFile(filename,'r') as f:
		f.extractall()# unzipping
	os.remove(filename) # removing zip file to save space


# Converting PFM to PNG 
filenames = glob.glob('*-perfect') # All unzip filenames
print('PFM to PNG conversion ...')
for filename in filenames:

	# Removing -sd.pfm since I am not interested in those
	sd_pfm_files = glob.glob(filename + '/*-sd.pfm')
	for sd_pfm_file in sd_pfm_files:
		os.remove(sd_pfm_file)

	# removing pgm files since I am not interested in those
	pgm_files = glob.glob(filename + '/*.pgm')
	for pgm_file in pgm_files:
		os.remove(pgm_file)
	# removing lightning images (E & L) since I am not interested in those
	lightning_file = glob.glob(filename + '/*1L.png')
	if(bool(lightning_file)):
		os.remove(lightning_file[0])
	lightning_file = glob.glob(filename + '/*1E.png')
	if(bool(lightning_file)):
		os.remove(lightning_file[0])

## Creation of the pickle
depth_imgs = []
rgb_imgs  = []

for filename in filenames:
	png_depth_files = glob.glob(filename + '/*.pfm')
	png_im_files = glob.glob(filename + '/im*.png')
	for png_depth_file, png_im_file  in zip(png_depth_files, png_im_files):
		# Adding depth file
		depth = cv2.imread(png_depth_file, cv2.IMREAD_UNCHANGED) # read as gray img
		depth[depth==np.inf] = 0
		# depth = cv2.normalize(depth,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
		cv2.imwrite(png_depth_file[:-4]+'.png',depth.astype(np.uint16))
		depth_imgs.append(depth)

		os.remove(png_depth_file)

		# Adding im file
		im = cv2.imread(png_im_file, 3) # read image
		# im = im / 255.0 # normalizing
		cv2.imwrite(png_im_file[:-4]+'.png',im)
		rgb_imgs.append(im)

# File lists are ready, let's save them into a pickle
# Saving depth images
with open('depth.pickle', "wb") as pickle_out:
	pickle.dump(depth_imgs, pickle_out)

# Saving images
with open('rgb.pickle', "wb") as pickle_out:
	pickle.dump(rgb_imgs, pickle_out)

# Done