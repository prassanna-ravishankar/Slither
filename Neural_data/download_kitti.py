import os, sys
import requests
import zipfile
import json
from PIL import Image
import numpy as np

def write_file(local_filename, url_to_download, cur_folder):
	with open(cur_folder + '/'+local_filename, "wb") as f:
		print ("Downloading %s" % local_filename)
		response = requests.get(url_to_download, stream=True)
		total_length = response.headers.get('content-length')

		if total_length is None: # no content length header
			f.write(response.content)
		else:
			dl = 0
			total_length = int(total_length)
			for data in response.iter_content(chunk_size=4096):
				dl += len(data)
				f.write(data)
				done = int(50 * dl / total_length)
				sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
				sys.stdout.flush()
			sys.stdout.write("\n")

folder_of_file = os.path.dirname(os.path.abspath(__file__))
folder_target = os.path.join(folder_of_file, "KittiSemantic")
if not os.path.exists(folder_target):
	os.mkdir(folder_target)

print ("1. Downloading Kitti")
if not os.path.exists(folder_target+"/kitti_semantic.zip"):
	write_file("kitti_semantic.zip","https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip", folder_target)

print("2. Unzipping")
f_obj = zipfile.ZipFile(folder_target+"/"+"kitti_semantic.zip",'r')
f_obj.extractall(folder_target)
f_obj.close()

print("3. Collecting File list")
training_json = dict()
testing_json = dict()
training_files = os.listdir(folder_target+"/training/image_2")
testing_files =os.listdir(folder_target+"/testing/image_2")


print ("4. Scanning through Annotations")
unique_annotations = set()
for filename in training_files:
	unique_in_this_image = set(np.unique(Image.open(
		os.path.join(folder_target, "training", "semantic", filename)
		)))
	unique_annotations = unique_annotations.union(unique_in_this_image)
print(":::unique annotations found as  {}".format(unique_annotations))
training_json['labels'] = ', '.join(str(x) for x in unique_annotations)

print ("5. dumping dataset index files")
open(folder_target+"/training.txt", "w").writelines([f + "\n" for f in training_files])
open(folder_target+"/testing.txt", "w").writelines([f + "\n" for f in testing_files])
training_json['filenames'] = training_files
training_json['image_folder'] = os.path.join(folder_target,"training","image_2")
training_json['annotation_folder'] = os.path.join(folder_target, "training", "semantic")
testing_json['filenames'] = testing_files
testing_json['image_folder'] = os.path.join(folder_target,"testing","image_2")
for k in training_json:
	print ((k, type(k)))
for k in testing_json:
	print ((k, type(k)))
json.dump(training_json, open(folder_target+"/train.json", "w"))
json.dump(testing_json, open(folder_target+"/test.json", "w"))

