import cv2 
import numpy as np
from argparse import ArgumentParser
from histogram_analysis import main as hist_main
from glob import glob
import os
import copy






def main():
	parser = ArgumentParser()
	parser.add_argument("-p", "--path", required = True, help = " path to folder containg images", type = str)
	args = parser.parse_args()
	path = args.path
	promotags = []
	if path[len(path)-1] != '/':
		path = path + '/' 
	
	filenames = glob(path + '*.jpg')

	for file in filenames:
		promotags.append(cv2.imread(file))
	promotags_copy = copy.copy(promotags)
	# print(promotags[0])
	index = range(0,len(promotags))

	grouping = hist_main(promotags_copy,index)
	# print(grouping)
	os.mkdir(path + "clusters")
	clusters_path = path + 'clusters/'

	for group, indexes in grouping.iteritems():
		count = 0
		os.mkdir(clusters_path + '_group' + str(group))
		new_path = clusters_path + '_group' + str(group) +'/'
		for idx in indexes:
			cv2.imwrite(new_path + str(count) + '.jpg', promotags[idx])
			count += 1



	



if __name__ == "__main__":
	main()
