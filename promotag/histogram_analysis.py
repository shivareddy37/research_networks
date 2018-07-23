import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import distance as dist


def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.4 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])

	# return the chi-squared distance
	return d

def comapare_hist(query_img, products):
	
	# initialize the index dictionary to store the image index
	# and corresponding histograms and the images dictionary
	# to store the images themselves
	index = {}
	images = {}
	results = []
	del results[:]
	for i in range(len(products)) :
		images[i] = cv2.cvtColor(products[i], cv2.COLOR_BGR2RGB)

		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update
		# the index
		hist = cv2.calcHist([products[i]], [0, 1, 2], None, [8, 8, 8],
			[0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		index[i] = hist

	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		# d = cv2.compareHist(index[0], hist, cv2.HISTCMP_CORREL)
		d1 = dist.cityblock(index[0], hist)
		# d2 = chi2_distance(index[0], hist)
		
		if d1  < 5.0  :
			results.append(k)
	index.clear()
	images.clear()	
	# print(results)
	return results





def main(products, prod_index):



	groups = {}
	group_number = 1
	
	
	
	
	while len(products) > 0:
		
		g_l = comapare_hist(products[0], products)
		l = []
		for k in g_l:
			l.append(prod_index[k])


		groups[group_number] = l
		group_number+=1
		
		

		for k in sorted (g_l, reverse = True):
			del products[k]
			del prod_index[k]

	# 	# print(prod_index)
	# m = 0 # max size of grouping needed for majority vote

	# for k,g in groups.iteritems():
	# 	if len(g) > m:
	# 		m = len(g)

	# count_max = 0 # how many times does that max size occour if more that=n one undecisive

	# for k,g in groups.iteritems():
	# 	if len(g) == m:
	# 		count_max += 1

	# if count_max ==1:
	# 	for k,g in groups.iteritems():
	# 		if len(g) == m:
	# 			return g

	# else:
	# 	return []	
	return groups		


		
		
	




if __name__ == "__main__":
	main()