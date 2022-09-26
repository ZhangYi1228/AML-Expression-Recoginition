# -*- coding: utf-8 -*-
    # get_uniform_map.py
    # 2015-7-7
    # github: https://github.com/michael92ht
    #__author__ = 'huangtao'

#The dictionary obtained by serializing and numbering the 58 eigenvalues of the uniform mode from small to large

#Find the 8 values whose binary representation rotates once
def circle(arr,values):
    for i in range(0,8):
        b=0
        sum=0
        for j in range(i,8):
            sum+=arr[j]<<b
            b+=1
        for k in range(0,i):
            sum+=arr[k]<<b
            b+=1
        values.append(sum)
#In the characteristic binary representation of the equivalent mode, 1 appears continuously. The simulation shows 1 to 7 1s in a row, and then finds the 8 values ​​of its rotation.
def get_from():
    f=open(r"test.txt",'w')
    values=[]
    for i in range(1,8):
        init_value=0
        init_length=8
        arr=[init_value]*init_length
        j=0
        while(j<i):
            arr[j]=1
            j+=1
        circle(arr,values)    
    values.sort()
    num=1
    map={}
    for v in values:
        map[v]=num
        num+=1
        f.write(str(v)+':'+str(map[v])+',')
       

if __name__ == '__main__':
    get_from()      
