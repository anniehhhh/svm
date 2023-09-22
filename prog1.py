questions = [
     ["is sky blue in colour?","yes","no","both a and b","none"],
     ["is a tree green in colour?","yes","no","both a and b","none"],
     ["is your dress pink in colour?","yes","no","both a and b","none"],
     ["is your eyes brown in colour?","yes","no","both a and b","none"],
     ["is your hair black in colour?","yes","no","both a and b","none"],
]
money=0
levels=[1000,2000,3000,5000,10000]
correct=["yes","yes","no","both a and b","none"]
for  i in range(0,len(questions)):
    question=questions[i]
    print(f"\n question for Rs.{levels[i]} is {question[0]}")
    print(f"a. {question[1]} b.{question[2]}")
    print(f"c. {question[3]} d.{question[4]}")
    ans=int(input("enter your answer (1-4)"))
    if(question[ans] == correct[i]):
        print(f"you won Rs.{levels[i]}!")
        money= money+levels[i]
    else:
        print(f"you lost! you will get Rs.{money}")
        break

print(f"thank you.")


