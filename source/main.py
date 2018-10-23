#Panoráma kép készítő alkalmazás
#Farkas Balázs P3L47H


# import references
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import constans as c
import logic2 as l


#Start
print("Start application: ", c.APPNAME)
imageNames=[]

#get files from path
try:
    print("Path: ",c.PATH)
    imageNames = [fn for fn in os.listdir(c.PATH)
            if any(fn.upper().endswith(ext) for ext in c.INCLUDED_EXTENSIONS)]
except:
    print("HIba a fájlok beolvasása közben.")

print(imageNames)
numberOfImages= len(imageNames)
images= [None] * numberOfImages
imageCounter=0

# Load imagefiles to array
for imageName in imageNames:
    fullImagePath=c.PATH+"\\"+imageName
    images[imageCounter]=cv2.imread(fullImagePath)
    images[imageCounter]=images[imageCounter][:,:,::-1] #BGR>>RGB
    imageCounter+=1
    print(imageCounter,": ",imageName)


fig, axes = plt.subplots(nrows = 2, ncols = numberOfImages)
fig.suptitle(c.FIGURE_TITLE, fontsize=16)

for ax,image in zip(axes.flat,images):
   ax.imshow(image)

# remove all axes from second row
for i in range(numberOfImages,1,-1):
    ax=axes[1,i-1]
    ax.remove()

# add axes in full width
gs = axes[0, 0].get_gridspec()
axbig = fig.add_subplot(gs[1, 0:])
ax=axes[1,0]
ax.remove()

fullImage=[None]
for image in images:
    fullImage=l.mergeTwoImage(fullImage,image,True)
    print("FullImage (X,Y): ",len(fullImage[0]),len(fullImage))

axbig.imshow(fullImage)


# set plt to full screen and show
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()
