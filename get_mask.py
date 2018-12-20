import os, glob
from ij import IJ, Prefs, ImagePlus, WindowManager as WM
from ij.plugin import ImageCalculator

#parameters
mini = [7.58,   127,    4.02]
maxi = [22.57,   130,   16.42]
filteri = ["pass","pass","pass"]
stackinds = ["L*", "a*", "b*"]
Prefs.blackBackground = False
savepath = "/mnt/myFolder/phil/scannerData/D&Dretry/"
newpath  = "/home/lpe/Desktop/slime_threshed/dnd_retry/"
#newpath = savepath.__add__("newfolder/")
print newpath




def action(filename):
	prefix = filename[:-4]
	newprefix = prefix.replace(savepath, newpath)
	print newprefix
	imp = IJ.openImage(filename)
	print filename
	ayo = IJ.run(imp, "Lab Stack", "")

	print ayo
	ic = ImageCalculator()

	IJ.run(ayo, "Stack to Images", "")

	for x in range(0, 3):
		IJ.selectWindow(stackinds[x])
		IJ.setThreshold(mini[x], maxi[x])
		IJ.run(ayo, "Convert to Mask", "method=Huang")
		if filteri[x] == "stop":
			IJ.run(ayo, "Invert", "")


	images = []  
	for id in WM.getIDList():  
		 images.append(WM.getImage(id))

	print images

	la = images[0]
	aa = images[1]
	ba = images[2]

	la.show()
	aa.show()
	ba.show()
	rum = ic.run("AND create", la, ba)

	rum.show()




	IJ.saveAs(rum, "PNG",newprefix +  "_lab_thresh.png")


	la.close()
	aa.close()
	ba.close()
	rum.close()



for filename in glob.glob("/mnt/myFolder/phil/scannerData/D&Dretry/*25p.png"):
	action(filename)
