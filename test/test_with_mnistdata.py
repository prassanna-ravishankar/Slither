import mnist
from pySlither import slither

print(mnist.temporary_dir())
print("Modifying training features")
images = mnist.train_images()
train_images_as_features = [images[i,:,:].reshape(-1) for i in range(0,images.shape[0])]
train_labels = mnist.train_labels()

print("Modifying testing features")
images = mnist.test_images()
test_images_as_features = [images[i,:,:].reshape(-1) for i in range(0,images.shape[0])]
test_labels = mnist.test_labels()

print ("Training")
my_slither = slither()
my_slither.setDefaultParams()
my_slither.loadData(train_images_as_features, train_labels)
my_slither.onlyTrain()
my_slither.loadData(test_images_as_features, test_labels)

print ("Testing")
res_prob = mnist.onlyTest()
res_clf = np.argmax(res_prob, axis=1)
print("I got : " + str(np.sum(res_clf == Y_test)) + "correct out of : " + str(len(Y_test)))
