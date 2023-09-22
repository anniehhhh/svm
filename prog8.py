def prog(s,c):
    a=0
    for i in range(len(s)): 
        if (s[i]==c):
            a=a+1
    return a
s=input("enter the string\n")
c=input("enter the character\n")
x=prog(s,c)
if(x!=0):
    print(x)
else:
    print("none")