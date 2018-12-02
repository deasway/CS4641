import csv

x = []
with open("winequality-white.csv", "rb") as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        line = row[0].split(";")
        x.append(line)
f.close()

with open("winequality-white-simplified.csv", "wb") as f:
    writer = csv.writer(f)
    for line in x:
        if int(line[-1]) < 4:
            line[-1] = 0
        elif int(line[-1]) < 7:
            line[-1] = 1
        else:
            line[-1] = 2
    writer.writerows(x)
f.close()
    
