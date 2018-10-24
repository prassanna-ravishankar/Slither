import os, argparse, sys
import requests
# for the dataset

folder_of_file = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Download Mnist Data');
parser.add_argument('--train', type=bool, help='Download Training set', default=True)
parser.add_argument('--test', type=bool, help='Download Testing set', default=True)
args = parser.parse_args()

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

if (args.train):
	print ("[Downloading the Training set]")
	train_url =  "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	write_file("train-images-idx3-ubyte.gz",train_url, folder_of_file)
	print (" ..... and some labels")
	train_label_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
	write_file("train-label-idx1-ubyte.gz",train_url, folder_of_file)

if args.test:
	print ("[Downloading the Testing set]")
	test_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
	test_label_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
	write_file("t10k-labels-idx1-ubyte.gz", test_label_url)
	print (" ..... and some labels")
	write_file("t10k-images-idx3-ubyte.gz", train_label_url)

print ("[TODO] Extract these into requireed folder")