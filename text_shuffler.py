import random 
 
# Open the file in read mode and read all the lines into a list 
with open('/Users/xiangchaolei/len/train.csv', 'r') as f: 
    lines = f.readlines()
 
# Shuffle the list of lines randomly 
random.shuffle(lines) 
 
# Open the same file in write mode and write the shuffled lines back to the file 
with open('/Users/xiangchaolei/len/train.csv', 'w') as f: 
    f.writelines(lines)