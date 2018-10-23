#Panoráma kép készítő alkalmazás
#Farkas Balázs P3L47H

# referenciák
import matplotlib.pyplot as plt
import cv2
import os
import constans as c
import logic2 as l

#Start
print("Start application: ", c.APPNAME)
imageNames=[]

#képek betöltése
try:
    fullPath=os.path.join(os.getcwd(),'source','test',c.TESTPATH)
    print("Path: ",fullPath)
    imageNames = [fn for fn in os.listdir(fullPath)
            if any(fn.upper().endswith(ext) for ext in c.INCLUDED_EXTENSIONS)]
except:
    print("Hiba a fájlok beolvasása közben.")

print(imageNames)
numberOfImages= len(imageNames)
images= [None] * numberOfImages
imageCounter=0

# Képfájlok betöltése tömbből
for imageName in imageNames:
    fullImagePath=os.path.join(fullPath,imageName)
    images[imageCounter]=cv2.imread(fullImagePath)
    images[imageCounter]=images[imageCounter][:,:,::-1] #BGR>>RGB
    imageCounter+=1
    print(imageCounter,": ",imageName)


fig, axes = plt.subplots(nrows = 2, ncols = numberOfImages)
fig.suptitle(c.FIGURE_TITLE, fontsize=16)

for ax,image in zip(axes.flat,images):
   ax.imshow(image)

# összes tengely eltávollítása a másodi sorból
for i in range(numberOfImages,1,-1):
    ax=axes[1,i-1]
    ax.remove()

# teljes szélességű tengely hozzáadása
gs = axes[0, 0].get_gridspec()
axbig = fig.add_subplot(gs[1, 0:])
ax=axes[1,0]
ax.remove()

fullImage=[None]
for image in images:
    fullImage=l.mergeTwoImage(fullImage,image,c.DEBUG_MODE)
    print("FullImage (X,Y): ",len(fullImage[0]),len(fullImage))

axbig.imshow(fullImage)

# teljes képernyős plotter beállítása
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()
print("Application ended.")