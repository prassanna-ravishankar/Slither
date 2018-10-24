import os, sys
import requests

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

print ("1. Downloading the model")
model_url = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel"
write_file("bvlc_googlenet.caffemodel",model_url, folder_of_file)

print ("2. Downloading the protxt")
protxt_url = "https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/bvlc_googlenet.prototxt"
write_file("bvlc_googlenet.prototxt", protxt_url, folder_of_file)

print ("3. Downloading ILSVRC2012 classes")
classes_url = "https://github.com/opencv/opencv/tree/3.4/samples/dnn/classification_classes_ILSVRC2012.txt"
write_file("classification_classes_ILSVRC2012.txt", classes_url, folder_of_file)

print ("-------Downloaded necessary resources-----------")