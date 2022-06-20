
file1 = open("word_in_1OOO_need_to_learn.txt") 
output = open("output2.txt","w")
 

lines = file1.readlines()

for i,line in enumerate(lines):
    try:
        output.write(lines[i].replace("\n","")+" -- "+lines[len(lines)//2+1+i ])
    except:
        break
