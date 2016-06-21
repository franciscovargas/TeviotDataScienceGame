import os
import numpy as np
import cv2

aListx = [];
aListy = [];

counter = 0;

path = '/afs/inf.ed.ac.uk/user/s14/s1433525/teviot/TeviotDataScienceGame/data/images/roof_images/'

for filename in os.listdir(path):

	if not filename.endswith('.jpg'): continue

	fullname = os.path.join(path, filename)

	counter = counter + 1

	image = cv2.imread(fullname)
	height = np.size(image, 0)
	width = np.size(image, 1)

	aListx.append(width)
	aListy.append(height)

print "Mean x: %f" % np.mean(aListx)
print "Standard Deviation x: %f" % np.std(aListx)
print "Max x: %f" % np.max(aListx)
print "Min x: %f\n" % np.min(aListx)

print "Mean y: %f" % np.mean(aListy)
print "Standard Deviation y: %f" % np.std(aListy)
print "Max y: %f" % np.max(aListy)
print "Min y: %f\n" % np.min(aListy)

print "Number of images: %d" % counter
