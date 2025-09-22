# String

""" fi_naem = "Jhon"
sc_name = "Key"

message = f"Hello! {fi_naem} {sc_name}!"
print(message.upper())
print(message.lower())
message2 = "     hello   !    "
print(message2+message)
print(message2.lstrip()+message)
print(message2.rstrip()+message)
print(message2.strip()+message)
WebOne = "https://hello"
print(WebOne.removeprefix("https:"))
print(WebOne.removesuffix("llo"))
print(WebOne)
WebOne = WebOne.removeprefix("https")
print(WebOne) """

# Num

""" NUM = 100_00_001
print(NUM)
num = 100_100
print(num)
print(2**3)
x, y, z = 0, 1, 2
print(x,y,z)
print(f"{x}and{y}and{z}") """


""" import this """

# List

""" list = ["b","c","q","a","d","s","e"]
listt = list
print(list)
print(list[3])
print(len(list))
list[1] = "f"
print(list)
list.append(3)
list.insert(0,13)
print(list)
del list[1]
print(list)
li = list.pop()
print(f"{li}   {list}")
li2 = list.pop(0)
print(f"{li2}   {list}")   # no it and not delete : wrong!
print(list)
list.remove("a")
print(listt)
listt.sort()
print(listt)
print(list) 
print(sorted(list))
print(list.reverse())
print(list) 
 """

# ListOperation

""" Object = ["math","chinese","eglish"]
for x in Object:
    print(f"I like {x}")
print("OHHHHH!")
for x in range(0,6):
    print(x)
nums = list(range(0,100,10))
print(nums)
list = []
for x in range(1,10,1):
    sq = x**2
    list.append(sq)
    list.append(x**2)
print(list)
print(min(list),max(list),sum(list))
nums = [x for x in range(1,10,1)]
print(nums)
nums = [x**2 for x in range(1,10,2)]
print(nums) """

# Slice

""" player = ["hello","my","name","is","zhang","gao","sheng"]
print(player[0:3])
player2 = player[0:3]
print(player2)
player2 = player
player2.append("HHHH")
print(player2) """

# Tuple

""" nums = (100,200,"hello",3000,1)
print(nums)
print(nums[1])
newNums = 1,2,100,"hello"
print(newNums)
newNums2 = (1,)
newNums3 = 1,
newNums4 = 1
print(f"{newNums2}  and  {newNums3}  and  {newNums4}")
nums = (100,500)
for x in nums:
    print(x) """

# If

""" Class = ["math","eglish","chinese","chemistry","art"]
for x in Class:
    if x == "math":
        print(f"I like {x}")
    if x == "eglish":
        print(f"OH ! {x}")
    if "chinese" in Class:
        print("ok")
nums = [x for x in range(1,100)]
for x in nums:
    if x>=10 and x<=50:
        print(f"{x} is in 10 and 50")
nums = []
if nums:
    print("Aaaa")
else:
    print("bbbbb") """

# Dictionary

""" person = {"name":"John",
          "sex":"man", 
          "age":18, 
          "grade":[100,98,96], 
          "test":{"test1":1, "test2":"hello"}, 
          "name":"Keta"
          }
print(person)
print(person["name"])
print(person["grade"])
print(person["grade"][1])
print(person["test"]["test1"])
newPerson = {}
newPerson["name"] = "Akasa"
newPerson["age"] = 17
print(newPerson)
for x in person:
    print(x)
    print(person[x])
del person["test"]
print(person)
x = person.get("house","no house")
print(x)
x = person.get("name","no house")
print(x)
x = person.get("house")
print(x)
for x in person:
    print(x)
    print(person[x])
for x,y in sorted(person.items()):
    print(f"the {x} and {y}")
print(person.items())
for x in person.keys():
    print(x)
for x in person.values():
    print(x)
key = person.keys();
value = person.values()
print(f"{key}\n{value}")
grade = {"math":100,
         "chinese":100,
         "english":120
         }
for x in set(grade.values()):
    print(x)
player = {"John","keta","masa"}
print(player)
for x in player:
    print(x)
aline = []
for x in range(1,31,1):
    newAlien = {"color":"yellow",
                "point":5,
                "speed":"slow"
                }
    aline.append(newAlien)
for x in aline[:5]:
    print(f"{x}") """

# Input and While

""" message = input("input a name")
print(message)
message = "input a num"
nums = input(message)
print(nums+"hello")
nums = int(nums)
print(nums+"hello")
num = 1
while num <=5:
    print(num)
    num += 1
flag = True
while flag:
    message = input()
    if message != "quit":
        print(message)
    else:
        break
x = 1
sum = 0
while x <= 100:
    if(x%2 == 0):
        sum += x
    if(x == 50):
        break
    x += 1
print(sum)
list = ["hello","hh","hell","hello"]
while list:
    tmp = list.pop()
    print(tmp)
while "hello" in list:
    list.remove("hello")
print(list)
list = {}
yes = True
while yes:
    name = input("your name : ")
    res = int(input("your goal : "))
    list[name]=res
    re = input("Yes or No : ")
    if(re == "No"):
        yes = False
print(list) """

# Function

""" def say():
    print("hello")
say()
def say(word):
    print(word)
say("Hello!")
def say(book):
    print(f"my favorite book is {book}")
say("Amazon")
def fun(x,y):
    x = int(x)
    y = int(y)
    print(x-y)
fun(x=3,y=4)
fun(x=4,y=3)
def fun(x,y):
    print(x+"    "+y)
fun("hello","1")
def sum(x, y):
    x = int(x)
    y = int(y)
    return x+y
print(sum(4,6))
def name(first_name,second_name):
    full_name = f"{first_name} {second_name}"
    return full_name
myName = name("John","Kim")
print(myName)
myName = name(first_name="Jion",second_name="Mary")
print(myName)
myName = name(second_name="hello",first_name="Jion")
print(myName)
def name(first_name, second_name,mid_name=""):
    if mid_name :
        full_name = f"{first_name} {second_name}"
    else:
        full_name = f"{first_name} {mid_name} {second_name}"
    return full_name
myName = name("Jon","kim","lee")
print(myName)
def name(first_name, mid_name="",second_name,): #Wrong Operation 
    if mid_name :
        full_name = f"{first_name} {second_name}"
    else:
        full_name = f"{first_name} {mid_name} {second_name}"
    return full_name
myName = name(first_name="Cookie",mid_name="session",second_name="hello")
print(myName)
def name(first_name,second_name):
    full_name = {"first" : first_name, "second" : second_name}
    return full_name
myName = name("John","Kim")
print(myName)
def person(first_name, second_name,age=None): # the number's normal is None
    person = {"first" : first_name, "second" : second_name}
    if age :
        person["age"] = int(age)
    return person
newPerson = person("John","Kim",13)
print(newPerson)
def city_country(city,country):
    s = f"{city}, {country}"
    return s
newCity = city_country("Santiago","Chile")
print(newCity)
def Bubble_Sort(list): # Bubble_Sort Demo
    lens = len(list)
    for i in range(0,lens-2,1):
        for j in range(0,lens-1-i,1):
            if(list[j]>list[j+1]):
                list[j], list[j+1] = list[j+1],list[j]
    return list
list = [5,1,3,8,2,9,4]
print(list)
Bubble_Sort(list[:]) # slice is the number transformation
print(list)
Bubble_Sort(list) # Adress Transformation
print(list)
def Printer(**x):
    print(x)
Printer(hh = "hello" ,la = " My" )
demo = {"hh" : "hello",
        "la" :  "my"}
import math
print(math.log(4,2)) """

# Class

""" try:
    while True:
        n = int(input())
        if n==0:
            print(0)
            continue
        li = []
        while n > 0:
            li.append(n%2)
            n//=2
        for i in range(len(li)-1, -1,-1):
            print(li[i],end='')
        print('')
except Exception as e:
    exit(0) """

""" try:
    while True:
        num = int(input())
        if num == 0:
            print(0)
            continue
        li = []
        while num > 0:
            li.append(num%2)
            num//=2
        sum = int(0)
        tmp = int(1)
        for i in range(len(li)-1,-1,-1):
            sum += li[i]*tmp
            tmp*=2
        print(sum)
except Exception as e:
    exit(0) """

""" n = int(input())
li = []
while n > 0:
    li.append(int(n%2))
    n//=2
while len(li)-1 < 40:
    li.append(int(0))
li[16:32].reverse()
ans1 = li[16:32]
li[0:16].reverse()
ans2 = li[0:16]
tmp = int(1)
sum = int(0)
for i in range(0,len(ans1),1):
    sum += li[i]*tmp
    tmp*=2
for i in range(0,len(ans2),1):
    sum += li[i]*tmp
    tmp*=2
print(ans1)
print(ans2)
print(sum) """

""" n = int(input().strip())
for i in range(1,n+1,1):
    a = int(input().strip())
    b = int(input().strip())
    x = int(0)
    while a**x <= b:
        x = x + 1
    print(x-1)
    if i != n:
        m = input() """

""" num = int(input().strip())
while num:
    msg = input().strip()
    n = int(input().strip())
    for i in range(0,n,1):
        words = list((input().strip().split(" ")))
        for j in range(0,len(words),1):
            print(words[j][::-1],end="")
            if j != len(words)-1:
                print(end=" ")
        print()
    print()
    num = num - 1 """

""" def reverse_words_in_line(line):  
    words = line.split()  
    reversed_words = [word[::-1] for word in words]    
    reversed_line = ' '.join(reversed_words)  
    return reversed_line  
  
M = int(input().strip())  
  
for i in range(M):
    msg = input().strip()
    n = int(input().strip())
    for j in range(n):
        line = input().strip()  
        print(reverse_words_in_line(line))  
    print() """
