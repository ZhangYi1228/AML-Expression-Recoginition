# -*- coding: utf-8 -*-
    # get_resolve_map.py


#Find the dictionary obtained by serializing and numbering the 36 eigenvalues of the rotation invariant mode from small to large

import string

#Take the minimum value of one rotation of a binary representation
def get_min_for_revolve(arr,values):
    h=[]
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
        h.append(sum)
    values.append(min(h))
    
if __name__ == '__main__':
    
    values=[]
    for i in range(0,256):  #Find the minimum value of the binary representation of 0~255 after one rotation
        b=bin(i)
        length=len(b)
        arr=[]
        j=length-1
        while(j>1):             #Constructs a list of binary representations
            arr.append(int(b[j]))
            j-=1
        for s in range(0,8-len(arr)):
            arr.append(0)
        get_min_for_revolve(arr,values)
        
    values.sort()           #construct dictionary
    map={}
    num=0
    f=open(r"test.txt",'w')
    for v in values:
        #if not map.has_key(v):
        if v not in map:
            map[v]=num
            num+=1
            f.write(str(v)+':'+str(map[v])+',')



        
        
