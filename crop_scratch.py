# @IOService io
# @UIService ui
# @ImageJ ij
#@OUTPUT Dataset output
#@ DatasetService ds
#@ OpService ops
import os, glob
from net.imagej.axis import Axes
from net.imglib2.util import Intervals
from ij import IJ
from net.imglib2.img.display.imagej import ImageJFunctions


# This function helps to crop a Dataset along an arbitrary number of Axes.
# Intervals to crop are specified easily as a Python dict.


def get_axis(axis_type):
    return {
        'X': Axes.X,
        'Y': Axes.Y,
        'Z': Axes.Z,
        'TIME': Axes.TIME,
        'CHANNEL': Axes.CHANNEL,
    }.get(axis_type, Axes.Z)

def crop(ops, data, intervals):
    """Crop along a one or more axis.
    Parameters
    ----------
    intervals : Dict specifying which axis to crop and with what intervals.
                Example :
                intervals = {'X' : [0, 50],
                             'Y' : [0, 50]}
    """

    intervals_start = [data.min(d) for d in range(0, data.numDimensions())]
    intervals_end = [data.max(d) for d in range(0, data.numDimensions())]

    for axis_type, interval in intervals.items():
        index = data.dimensionIndex(get_axis(axis_type))
        intervals_start[index] = interval[0]
        intervals_end[index] = interval[1]

    intervals = Intervals.createMinMax(*intervals_start + intervals_end)

    output = ops.run("transform.crop", data, intervals, True)

    return output

#imp.setRoi(1348,2372,1202,1144);
#imp.setRoi(120,2372,1202,1144);
#imp.setRoi(140,1236,1202,1144);
#imp.setRoi(1348,1316,1202,1144);
#imp.setRoi(1348,48,1202,1144);
#imp.setRoi(160,96,1202,1144);
# Define the intervals to be cropped
intervals = {'X': [212, 1304],
    'Y': [136, 1268]}

intervals1 = {'X': [1350, 2424],
    'Y': [110, 1236]}

intervals2 = {'X': [212, 1308],
    'Y': [1390, 2350]}

intervals3 = {'X': [208, 1300],
    'Y': [2396, 3500]}

intervals4 = {'X': [1360,2544],
    'Y': [1248, 2348]}

intervals5 = {'X': [1364, 2504],
    'Y': [2356, 3460]}
stackinds = ["1left", "1right","2left", "2right","3left", "3right"]
interval_dict = [intervals,intervals1,intervals2,intervals3,intervals4,intervals5]
print interval_dict[1]
savepath = "/home/lpe/Desktop/slime_threshed/dnd_retry/"
newpath  = "/home/lpe/Desktop/slime_threshed/dnd_retry_cropped/"





import glob
for cells in glob.glob('/home/lpe/Desktop/slime_threshed/dnd_retry/*25p*'):
	prefix = cells.split("/")
	print prefix
	blarp = prefix[6]
	print blarp
	nombre = blarp[:-4]
	print nombre
#	newprefix = prefix.replace(savepath, newpath)

#Crop the image 
	for cropregion in range(0,len(interval_dict)):
		image = ij.io().open(cells)
		intervals = interval_dict[cropregion]
		output = crop(ops, image, intervals)
		impThresholded=ImageJFunctions.wrap(output, "wrapped")
		print impThresholded

		IJ.saveAs(impThresholded, "PNG",newpath + stackinds[cropregion] +nombre + ".png")
		impThresholded.close() 

#output = crop(ops, cells, intervals2)
#impThresholded=ImageJFunctions.wrap(output, "wrapped")
#print impThresholded
#IJ.saveAs(impThresholded, "PNG",newprefix +  "d2_ntingdishes-1532023981-25.png")
#impThresholded.close()

# Create output Dataset
#output = ds.create(output)#output.close()
