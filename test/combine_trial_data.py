import pandas as pd 
import numpy as np 

import sys

ZERO_REPLACE_VAL = 100

INPUT_SIZE = int(sys.argv[1])

Ntrials = int(sys.argv[2])


dfs = []
for i in range(Ntrials):
    df = pd.read_csv('train_times_%i_%s.csv'%(i,INPUT_SIZE))


    dfs.append(df)

train_df = pd.concat(dfs,axis=0)

train_df = train_df.replace(0, ZERO_REPLACE_VAL)
print(train_df)

dfs = []
for i in range(Ntrials):
    df = pd.read_csv('test_times_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

test_df = pd.concat(dfs,axis=0)

test_df = test_df.replace(0, ZERO_REPLACE_VAL)
print(test_df)


train_df_means = train_df.mean(axis=0)
train_df_std = train_df.std(axis=0)/Ntrials

test_df_means = test_df.mean(axis=0)
test_df_std = test_df.std(axis=0)/Ntrials

print('test_df_std')
print(test_df_std)


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np


labels = ['tt4','tt5','tt6','tt7','tt8s']


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_df_means, width, label='Train', yerr=train_df_std)
rects2 = ax.bar(x + width/2, test_df_means, width, label='Test', yerr=test_df_std)

ax.set_ylabel('Time to x')
ax.set_title('Time to Finding New Basic Block Transitions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('img/2ladder_train_test_times_%s.png'%INPUT_SIZE)
# plt.show()
plt.clf()




ZERO_REPLACE_VAL = 500

dfs = []
for i in range(Ntrials):
    df = pd.read_csv('train_edits_until_x_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

train_df = pd.concat(dfs,axis=0)

train_df = train_df.replace(0, ZERO_REPLACE_VAL)
print(train_df)

dfs = []
for i in range(Ntrials):
    df = pd.read_csv('test_edits_until_x_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

test_df = pd.concat(dfs,axis=0)

test_df = test_df.replace(0, ZERO_REPLACE_VAL)
print(test_df)



dfs = []
for i in range(Ntrials):
    df = pd.read_csv('random_edits_until_x_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

random_df = pd.concat(dfs,axis=0)

random_df = random_df.replace(0, ZERO_REPLACE_VAL)
print(random_df)



dfs = []
for i in range(2):
    df = pd.read_csv('determ_edits_until_x_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

determ_df = pd.concat(dfs,axis=0)

determ_df = determ_df.replace(0, ZERO_REPLACE_VAL)
print(determ_df)


train_df_means = train_df.mean(axis=0)
train_df_std = train_df.std(axis=0)/Ntrials

test_df_means = test_df.mean(axis=0)
test_df_std = test_df.std(axis=0)/Ntrials

random_df_means = random_df.mean(axis=0)
random_df_std = random_df.std(axis=0)/Ntrials

determ_df_means = determ_df.mean(axis=0)
determ_df_std = determ_df.std(axis=0)/Ntrials

print('test_df_std')
print(test_df_std)


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np


labels = ['et4','et5','et6','et7','et8s']


x = np.arange(len(labels))  # the label locations
width = 0.35/2  # the width of the bars

fig, ax = plt.subplots()

rects2 = ax.bar(x - 3*width/2, train_df_means, width, label='Train', color='b', yerr=train_df_std)
rects3 = ax.bar(x - 1*width/2, test_df_means, width, label='Test', color='r', yerr=test_df_std)
rects1 = ax.bar(x + 1*width/2, random_df_means, width, label='Random', color='g', yerr=random_df_std)
rects1 = ax.bar(x + 3*width/2, determ_df_means, width, label='Determ', color='k', yerr=determ_df_std)

ax.set_ylabel('Edits to x')
ax.set_title('# edits to Finding New Basic Block Transitions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('img/2ladder_train_test_random_edits_until_%s.png'%INPUT_SIZE)
plt.clf()






################## edit cumulative plots


dfs = []
for i in range(Ntrials):
    df = pd.read_csv('train_transitions_per_edit_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

train_df = pd.concat(dfs,axis=0)

# train_df = train_df.replace(0, ZERO_REPLACE_VAL)
print(train_df)

dfs = []
for i in range(Ntrials):
    df = pd.read_csv('test_transitions_per_edit_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

test_df = pd.concat(dfs,axis=0)

# test_df = test_df.replace(0, ZERO_REPLACE_VAL)
print(test_df)

dfs = []
for i in range(Ntrials):
    df = pd.read_csv('random_transitions_per_edit_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

random_df = pd.concat(dfs,axis=0)


dfs = []
for i in range(2):
    df = pd.read_csv('determ_transitions_per_edit_%i_%s.csv'%(i,INPUT_SIZE))

    dfs.append(df)

determ_df = pd.concat(dfs,axis=0)

# random_df = random_df.replace(0, ZERO_REPLACE_VAL)
print(determ_df)



train_data = train_df.to_numpy()
test_data = test_df.to_numpy()
random_data = random_df.to_numpy()
determ_data = determ_df.to_numpy()[:,:random_data.shape[1]]

print(random_data)

# for eh in train_data:
#     plt.plot(eh,color='b',alpha=0.5, label='Train')

# for eh in test_data:
#     plt.plot(eh,color='r',alpha=0.5, label='Test')

# for eh in random_data:
#     plt.plot(eh,color='g',alpha=0.5, label='Random')

# for eh in determ_data:
#     plt.plot(eh,color='k',alpha=0.5, label='Determ')

# for eh in train_data:
plt.plot(train_data.mean(axis=0),color='b',alpha=1.0, label='Train')
plt.fill_between(np.arange(train_data.shape[1]), train_data.mean(axis=0) - train_data.std(axis=0)/Ntrials, train_data.mean(axis=0) + train_data.std(axis=0)/Ntrials, color='b', alpha=0.3)

# for eh in test_data:
plt.plot(test_data.mean(axis=0),color='r',alpha=1.0, label='Test')
plt.fill_between(np.arange(train_data.shape[1]), test_data.mean(axis=0) - test_data.std(axis=0)/Ntrials, test_data.mean(axis=0) + test_data.std(axis=0)/Ntrials, color='r', alpha=0.3)

# for eh in random_data:
plt.plot(random_data.mean(axis=0),color='g',alpha=1.0, label='Random')
plt.fill_between(np.arange(random_data.shape[1]), random_data.mean(axis=0) - random_data.std(axis=0)/Ntrials, random_data.mean(axis=0) + random_data.std(axis=0)/Ntrials, color='g', alpha=0.3)

# for eh in determ_data:
plt.plot(determ_data.mean(axis=0),color='k',alpha=1.0, label='Determ')
plt.fill_between(np.arange(determ_data.shape[1]), determ_data.mean(axis=0) - determ_data.std(axis=0)/Ntrials, determ_data.mean(axis=0) + determ_data.std(axis=0)/Ntrials, color='k', alpha=0.3)


plt.ylabel('Transitions found')
plt.xlabel('Total edits made')
plt.title('# edits to Finding New Basic Block Transitions')
plt.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('img/2ladder_train_test_random_edits_over_time_%s.png'%INPUT_SIZE)