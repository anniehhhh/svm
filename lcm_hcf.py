def lcm(x,y):
    if(x>y):
        g=x
    else:
        g=y
    while(True):
        if(g%x==0 and g%y==0):
            lcm=g
            break
        g=g+1
    return lcm
x=int(input())
y=int(input())
a=lcm(x,y)
print (a)