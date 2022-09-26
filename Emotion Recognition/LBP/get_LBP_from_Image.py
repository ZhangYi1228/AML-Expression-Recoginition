
# import the necessary packages
import numpy as np
import cv2
from PIL import Image
from pylab import*

class LBP:
    def __init__(self):
        #revolve_map is a dictionary obtained by serializing and numbering the 36 eigenvalues of the rotation invariant mode from small to large
        self.revolve_map={0:0,1:1,3:2,5:3,7:4,9:5,11:6,13:7,15:8,17:9,19:10,21:11,23:12,
                          25:13,27:14,29:15,31:16,37:17,39:18,43:19,45:20,47:21,51:22,53:23,55:24,
                          59:25,61:26,63:27,85:28,87:29,91:30,95:31,111:32,119:33,127:34,255:35}
        #uniform_map is a dictionary obtained by serializing and numbering the 58 eigenvalues of the equivalent mode from small to large
        self.uniform_map={0:0,1:1,2:2,3:3,4:4,6:5,7:6,8:7,12:8,
                          14:9,15:10,16:11,24:12,28:13,30:14,31:15,32:16,
                          48:17,56:18,60:19,62:20,63:21,64:22,96:23,112:24,
                          120:25,124:26,126:27,127:28,128:29,129:30,131:31,135:32,
                          143:33,159:34,191:35,192:36,193:37,195:38,199:39,207:40,
                          223:41,224:42,225:43,227:44,231:45,239:46,240:47,241:48,
                          243:49,247:50,248:51,249:52,251:53,252:54,253:55,254:56,
                          255:57}

        
     #Load the image and convert it to a grayscale image to obtain the pixel information of the grayscale image of the image
    def describe(self,image):
        image_array=np.array(Image.open(image).convert('L'))
        return image_array
    
    #Image LBP original feature calculation algorithm: compare the pixel at the specified position of the image with the surrounding 8 pixels
    #Points larger than the center pixel are assigned a value of 1, and points smaller than the center pixel are assigned a value of 0, and the resulting binary sequence is returned.
    def calute_basic_lbp(self,image_array,i,j):
        sum=[]
        if image_array[i-1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        return sum
    
    #Get the binary sequence for continuous circular rotation to get the smallest decimal value of the new binary sequence
    def get_min_for_revolve(self,arr): 
        values=[]
        circle=arr
        circle.extend(arr)
        for i in range(0,8):
            j=0
            sum=0
            bit_num=0
            while j<8:
                sum+=circle[i+j]<<bit_num
                bit_num+=1
                j+=1
            values.append(sum)
        return min(values)

    #Get the number of 1's in binary for the value r
    def calc_sum(self,r):
        num=0
        while(r):
            r&=(r-1)
            num+=1
        return num

    #Get the LBP raw mode features of the image
    def lbp_basic(self,image_array):
        basic_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute_basic_lbp(image_array,i,j)
                bit_num=0
                result=0
                for s in sum:
                    result+=s<<bit_num
                    bit_num+=1
                basic_array[i,j]=result
        return basic_array

   #Obtain the LBP rotation-invariant pattern features of an image
    def lbp_revolve(self,image_array):
        revolve_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute_basic_lbp(image_array,i,j)
                revolve_key=self.get_min_for_revolve(sum)
                revolve_array[i,j]=self.revolve_map[revolve_key]
        return revolve_array

  #Get LBP uniform Pattern Features of an Image
    def lbp_uniform(self,image_array):
        uniform_array=np.zeros(image_array.shape, np.uint8)
        basic_array=self.lbp_basic(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]

        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic_array[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic_array[i,j]^k
                 num=self.calc_sum(xor)
                 if num<=2:
                     uniform_array[i,j]=self.uniform_map[basic_array[i,j]]
                 else:
                     uniform_array[i,j]=58
        return uniform_array
    
    #Obtain the LBP rotation-invariant nuniform mode features of an image
    def lbp_revolve_uniform(self,image_array):
        uniform_revolve_array=np.zeros(image_array.shape, np.uint8)
        basic_array=self.lbp_basic(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic_array[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic_array[i,j]^k
                 num=self.calc_sum(xor)
                 if num<=2:
                     uniform_revolve_array[i,j]=self.calc_sum(basic_array[i,j])
                 else:
                     uniform_revolve_array[i,j]=9
        return uniform_revolve_array

    #Plots an image grayscale normalized statistical histogram of the specified dimension and range
    def show_hist(self,img_array,im_bins,im_range):
        hist = cv2.calcHist([img_array],[0],None,im_bins,im_range)
        hist = cv2.normalize(hist,None).flatten()
        plt.plot(hist,color = 'r')
        plt.xlim(im_range)
        plt.show()
        
    #Plot the normalized statistical histogram of the original LBP features of the image
    def show_basic_hist(self,img_array):
        self.show_hist(img_array,[256],[0,256])
        
    #Plotting normalized statistical histograms of image rotation-invariant LBP features
    def show_revolve_hist(self,img_array):
        self.show_hist(img_array,[36],[0,36])

    #Plotting normalized statistical histograms of image-equivalent pattern LBP features
    def show_uniform_hist(self,img_array):
        self.show_hist(img_array,[60],[0,60])
        
    #Plotting normalized statistical histograms of image rotation-invariant equivalent mode LBP features
    def show_revolve_uniform_hist(self,img_array):
        self.show_hist(img_array,[10],[0,10])

    #display image
    def show_image(self,image_array):
        cv2.imshow('Image',image_array)
        cv2.waitKey(0)

if __name__ == '__main__':
    image = r"test.jpg";
    lbp=LBP()
    image_array=lbp.describe(image)
    print(image_array)
    #Obtain the original LBP features of the image, and display its statistical histogram and feature image
    basic_array=lbp.lbp_basic(image_array)
    print(basic_array)
    lbp.show_basic_hist(basic_array)
    lbp.show_image(basic_array)

    #Obtain image rotation-invariant LBP features and display their statistical histograms and feature images
    revolve_array=lbp.lbp_revolve(image_array)
    #revolve_array = ((revolve_array+image_array)/2).astype('uint8')
    lbp.show_revolve_hist(revolve_array)
    lbp.show_image(revolve_array)

    #Obtain image-equivalent pattern LBP features and display their statistical histograms and feature images
    uniform_array=lbp.lbp_uniform(image_array)
    lbp.show_uniform_hist(uniform_array)
    lbp.show_image(uniform_array)

    
    #Obtain image rotation-invariant equivalent mode LBP features, and display its statistical histogram and feature image
    resolve_uniform_array=lbp.lbp_revolve_uniform(image_array)
    lbp.show_revolve_uniform_hist(resolve_uniform_array)
    lbp.show_image(resolve_uniform_array)
