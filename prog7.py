n=input('enter the range n')
c=0
for i in range(0,n):
    c=0
    if i==1 or i==0:
        continue
    for j in range(2,i):
        if i%j==0 :
            c=c+1
    if c==0:
        print(i)