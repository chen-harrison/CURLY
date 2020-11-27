import torch
from model_mrn import MRN
from contact_dataset import contactDataset


num_epochs = 2
batch_size = {"train": 24, "test": 4}
stack_size = 10
task_name_list = ['rf', 'lf', 'rb', 'lb']
root = '/home/harrison/Documents/CURLY/multitask/data/'

# list of paths for training and testing .mat files
paths = {'train': [root + '08292020_trial1.mat', root + '08292020_trial2.mat'],
            'test':  [root + '08292020_trial2.mat']}

# set up datasets and dataloaders
dsets = {'train': [], 'test': []}
for path in paths['train']:
    dsets['train'].append(contactDataset(path, stack_size))
for path in paths['test']:
    dsets['test'].append(contactDataset(path, stack_size))

# num_workers???
dloaders = {'train': [], 'test': []}
for dset in dsets['train']:
    dloaders['train'].append(torch.utils.data.DataLoader(dset, batch_size=batch_size["train"], shuffle=True, num_workers=1))
for dset in dsets['test']:
    dloaders['test'].append(torch.utils.data.DataLoader(dset, batch_size=batch_size["test"], shuffle=True, num_workers=1))   

model = MRN(device='cpu')

for epoch in range(num_epochs):
    for loader in dloaders['train']:
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            labels_list = torch.split(labels, 1, dim=1)
            model.optimize(inputs, labels_list)


    

    