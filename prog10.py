lst=[]
temp=[]
n=int(input("enter the number of sublists to be formed\n"))
for i in range(n):
    x=int(input("enter the number of elements in a sublist\n"))
    for j in range(x):
        temp.append(int(input()))
    lst.append(temp)    
    temp=[]
average=[]
for y in lst:
    if (len(y) != 0):
        average.append(sum(list(y))/len(y))
    else:
        average.append(0)
print(average)   
sum1=0
for z in average:
    sum1=sum(list(y))
average.remove(0)
total=sum1/len(average)
print(total) 

