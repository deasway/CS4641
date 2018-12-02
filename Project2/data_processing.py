import csv

x = []
with open("out2.txt","r") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(row)
        

f.close()

with open("data.csv","w") as f:
    writer = csv.writer(f)
    writer.writerows(x[:])

f.close()
                 
                        
