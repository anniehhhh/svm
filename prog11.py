lst=[]
temp=[]
lst2=[[0,0,0],[0,0,0],[0,0,0]]
n=int(input("enter the number of sublists to be formed n\n"))
for i in range(n):
    m=int(input("enter the number of elements in a sublist m\n"))
    for j in range(m):
        temp.append(int(input()))
    lst.append(temp)    
    temp=[]
for x in range(n):
    for y in range(len(lst[x])):
        lst2[y][x]=lst[x][y]
print(lst2)