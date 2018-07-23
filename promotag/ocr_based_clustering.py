import cv2
from tag_reader.tag_read import *
from glob import glob
from argparse import ArgumentParser
import os
import traceback


tag_reader = TagReaderGCPVision()

## fuction returns a bool based on the fact that it detects Sale or not.
def is_sale_tag(im):
	
	try:
		res1 = tag_reader.detect_single_cv(im, True)
		res2 = tag_reader.detect_single_cv(im, False)


		if res1 is not None and res2 is not None:
			result = res1.ocr_dump + res2.ocr_dump
			print(result)
			#result = str(result)
			#result = result.split('\n')
			#result = [x.upper() for x in result]

			if 'SALE' in result:
				return True
			else:
				return False
		else:
			return False
	except Exception:
		traceback.print_exc()
		return False

def gen_images(filenames):
	for filename in filenames:
		yield cv2.imread(filename)



def main(path, filenames):
	
	sale_tag_count = 1
	nonsale_tag_count = 1

	#  making directories 
	if not os.path.exists(path + 'sales_tags/'):
		os.makedirs(path + 'sales_tags/')
	if not os.path.exists(path + 'non_sales_tags/'):
		os.makedirs(path + 'non_sales_tags/')

	images = gen_images(filenames)
	count = 0
	while True: 
		try: 
			count += 1
			image = next(images)

			if is_sale_tag(image):
				print("was sales tag")
				cv2.imwrite(path + 'sales_tags/' + str(sale_tag_count) + '.jpg', image)
				sale_tag_count += 1
			else:
				print("was not sales tag")
				image_path = path + 'non_sales_tags/' + str(nonsale_tag_count) + '.jpg'
				# print(image_path)
				cv2.imwrite(image_path, image)
				nonsale_tag_count += 1
		except StopIteration:
			break

	print(count)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-p", "--path", required = True, help = " path to folder containg images", type = str)
	args = parser.parse_args()
	path = args.path
	if path[len(path)-1] != '/':
		path = path + '/' 
	promotags = []
	filenames = glob(path + '*.jpg')
	
	# for file in filenames:
	# 	promotags.append(cv2.imread(file))

	main(path , filenames)

