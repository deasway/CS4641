from sklearn import tree
from sklearn import preprocessing
import csv

#---------------LABEL PROCESSING----------------------
job_type = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"]
marital_status = ["married","divorced","single"]
education = ["unknown","secondary","primary","tertiary"]
credit_default = ["yes","no"]
loan = ["yes","no"]
contact_type = ["unknown","telephone","cellular"]
month_of_contact = ["jan","feb", "mar","apr","may","jun","jul","aug","sep","oct","nov", "dec"]
last_market_campaign_outcome = ["unknown","other","failure","success"]
binary = ["yes","no"]

le_job = preprocessing.LabelEncoder()
le_marital = preprocessing.LabelEncoder()
le_education = preprocessing.LabelEncoder()
le_contact = preprocessing.LabelEncoder()
le_month = preprocessing.LabelEncoder()
le_campaign = preprocessing.LabelEncoder()
le_binary = preprocessing.LabelEncoder()

le_job.fit(job_type)
le_marital.fit(marital_status)
le_education.fit(education)
le_contact.fit(contact_type)
le_month.fit(month_of_contact)
le_campaign.fit(last_market_campaign_outcome)
le_binary.fit(binary)
#---------------------------------------------------------

x = []
y = []
with open('bank-full.csv', 'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        line = row[0].replace('"', "").split(";")
        y.append(list(line))
        line.pop(14) #remove 9, 10, 11, 15 from list as pre-pruning
        line.pop(10)
        line.pop(9)
        line.pop(8)
        x.append(line)

for entry in x:
    entry[1] = le_job.transform([entry[1]])[0]
    entry[2] = le_marital.transform([entry[2]])[0]
    entry[3] = le_education.transform([entry[3]])[0]
    entry[4] = le_binary.transform([entry[4]])[0]
    entry[6] = le_binary.transform([entry[6]])[0]
    entry[7] = le_binary.transform([entry[7]])[0]
    entry[11] = le_campaign.transform([entry[11]])[0]
    entry[12] = le_binary.transform([entry[12]])[0]
# for entry in y:
#     entry[1] = le_job.transform([entry[1]])[0]
#     entry[2] = le_marital.transform([entry[2]])[0]
#     entry[3] = le_education.transform([entry[3]])[0]
#     entry[4] = le_binary.transform([entry[4]])[0]
#     entry[6] = le_binary.transform([entry[6]])[0]
#     entry[7] = le_binary.transform([entry[7]])[0]
#     entry[8] = le_contact.transform([entry[8]])[0]
#     entry[10] = le_month.transform([entry[10]])[0]
#     entry[15] = le_campaign.transform([entry[15]])[0]
#     entry[16] = le_binary.transform([entry[16]])[0]
f.close()

with open('bank-data.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(x[:])
f.close()

# with open('bank-data-extra.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(y[:])
# f.close()

full = []
with open("bank-additional-full.csv", "rb") as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        line = row[0].replace('"', "").split(";")
        full.append(line)
f.close()

job_type = ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"]
marital_status = ["divorced","married","single","unknown"]
education = ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"]
tertiary = ["no", "yes", "unknown"]
contact_type = ["telephone","cellular"]
month_of_contact = ["jan","feb", "mar","apr","may","jun","jul","aug","sep","oct","nov", "dec"]
day_of_week = ["mon","tue","wed","thu","fri"]
last_market_campaign_outcome = ["failure","nonexistent","success"]

le_job = preprocessing.LabelEncoder().fit(job_type)
le_marital = preprocessing.LabelEncoder().fit(marital_status)
le_education = preprocessing.LabelEncoder().fit(education)
le_tertiary = preprocessing.LabelEncoder().fit(tertiary)
le_contact = preprocessing.LabelEncoder().fit(contact_type)
le_month = preprocessing.LabelEncoder().fit(month_of_contact)
le_day = preprocessing.LabelEncoder().fit(day_of_week)
le_campaign = preprocessing.LabelEncoder().fit(last_market_campaign_outcome)
for entry in full:
    entry[1] = le_job.transform([entry[1]])[0]
    entry[2] = le_marital.transform([entry[2]])[0]
    entry[3] = le_education.transform([entry[3]])[0]
    entry[4] = le_tertiary.transform([entry[4]])[0]
    entry[5] = le_tertiary.transform([entry[5]])[0]
    entry[6] = le_tertiary.transform([entry[6]])[0]
    entry[7] = le_contact.transform([entry[7]])[0]
    entry[8] = le_month.transform([entry[8]])[0]
    entry[9] = le_day.transform([entry[9]])[0]
    entry[14] = le_campaign.transform([entry[14]])[0]
    entry[20] = le_binary.transform([entry[20]])[0]

with open("bank-extra.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(full[:])
f.close()



# with open('bank-training-extra.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(y[:31500])
# f.close()


# with open('bank-training2.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(x[6300:12600])
# f.close()

# with open('bank-training3.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(x[12600:18900])
# f.close()

# with open('bank-training4.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(x[18900:25200])
# f.close()

# with open('bank-training5.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(x[25200:31500])
# f.close()

# with open('bank-testing.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(x[31500:])
# f.close()
# with open('bank-testing-extra.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(y[31500:])
# f.close()







