d=dict()
n=int(input("enter the number of terms"))
for i in range(n):
    key=int(input("enter exponent"))
    value=int(input("enter value"))
    d.update({key:value})
print (d)
def diff(d):
    dp = {}
    for expo,val in d.items():
        if expo>0:
            dp[expo-1] = expo*val
    return dp
print(diff(d))