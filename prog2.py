word = input("enter the string")
strwrd=[]
strwrd = word.split(" ")
coding= False
if(coding):
    nwrd=[]
    for wrd in strwrd:
        if(len(wrd)>=3):
            s1="spt"
            s2="dfg"
            s3= s1+ wrd[1:]+wrd[0] +s2
            nwrd.append(s3)
            # print(nwrd.join(" "))
        else:
            nwrd.append(wrd[::-1])
    print(" ".join(nwrd))
else:
    nwrd=[]
    for wrd in strwrd:
        if(len(wrd)>=3):
            s=wrd[3:-3]
            s=s[-1]+s[:-1]
            nwrd.append(s)
        else:
            nwrd.append(wrd[::-1])
    print(" ".join(nwrd))




#practice....
word = input("enter the string")
print(word[1:])
print(word[:-1])
print(word[::-1])
print(word[:-1]+ word[0])
print(word[-1]+ word[:-1])
print(word[-1]+ word[1:])