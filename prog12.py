s=input("enter the string")
d=dict()
for i in s:
    if i not in d:
        d[i]=1
    else:
        d[i] +=1
print(d)


