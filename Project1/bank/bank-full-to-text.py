import csv

x = []
with open("bank-data-extra.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(row[:])
f.close()

with open("bank-data.txt", "w") as f:
    for instance in x:
        for i in range(len(instance)):
            if i != len(instance) - 1:
                f.write(instance[i] + ",")
            else:
                f.write(instance[i])
        f.write("\n")
            
f.close()
