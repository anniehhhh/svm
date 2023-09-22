a=0
b=1
print("fibonacci series is\n")
print(a)
print(b)
for i in range(2,6):
    c=a+b
    a=b
    b=c
    print(c)
print("the reverse is\n")
print(b)
print(a)
for i in range(2,6):
    temp=b-a
    print(temp)
    b=a
    a=temp
    