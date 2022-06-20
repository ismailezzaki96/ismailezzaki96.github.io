#usr/bin/env python


input_file = open("6FxXu86fl5mQwo7lEyDS5hG9NTc.txt")

output_file = open("output.txt","w")
output_file1 = open("output1.txt","w")

def replace_word_type(line):
    split_letter= "__"
    types_list = ["v", "adj", "adv", "conj", "det", "intj", "nadj", "prep", "pro","nf", "nm", "pl"]
    for typ in types_list:
        line=line.replace("("+typ+")" , split_letter).replace(" "+typ , split_letter).replace(","+typ+" " , split_letter).replace(typ+"," , split_letter).replace("(f)",split_letter)
    
    line=line[:line.find("_")]+" -- "+ line[line.rfind("_")+1:]
    return line

lines = input_file.readlines()

for i,line in enumerate(lines) :
    if "|" in line and line[1].isnumeric():
        continue
    if line[:line.find(" ")].isnumeric() :
        line= line[line.find(" ")+1:]
        line= replace_word_type(line)
        output_file.write(line) 
        
    if line[0] == "*":
        line=line.replace("* ","")
        next_line = lines[i+1]
        if "|" not in next_line:
            line = line.replace("\n"," ")+ next_line
        output_file.write(line)
