import cv2 as cns


imag = cns.imread('Photos/park.jpg')

# 1. converting the bgr image to greyscale image
gry = cns.cvtColor(imag,cns.COLOR_BGR2GRAY)      # the source imag(given)tobeconverted

#cns.imshow('p',imag)
cns.imshow('g',gry)
  
print(imag.shape)        # returns tuple of (height,width,channels) here in channel willprint 3,asthere_are3colors
                        # the numpy will return order in h,w,c but opencv excepts the order in width,height , also in opencv functionsthatweare_usingsameorderfollows                                            


# 2. blurring images
blr = cns.GaussianBlur(imag,(3,3),cns.BORDER_DEFAULT)     #more the (v1,v1) more the blur ,the kernelsize must be odd and itdefine_theare_overwhich_blurrisdefined
cns.imshow('g',blr)

# 3. Edge Cascade
cad = cns.Canny(imag,150,150)                  # min&max threshold_values
cns.imshow('c',cad)

# 4. Resizing the image
resize= cns.resize(imag,(500,750) , interpolation=cns.INTER_CUBIC)
cns.imshow('k',resize)

# 5. Cropping image
crop = imag[200:550 , 150:420]   # crop from x1tox2 and y1 to y2`` from height to width
cns.imshow('c',crop)

cns.waitKey(0)
 
