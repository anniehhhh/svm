def num(n):
    if(n==0):
        return "0"
    elif(n==1):
        return "1"
    else:
        return (n%2) + num(n/2)

a=num(5)
print(a)
    