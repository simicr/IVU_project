import os
import shutil
import json
import random
from sklearn.model_selection import train_test_split  

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

class DataManager():

    def __init__(self):
        super()
        self.cwd = os.getcwd()
        self.ok = False
        self.test_dir = os.path.join('archive', 'asl_alphabet_test', 'asl_alphabet_test')
        self.train_dir = os.path.join('archive', 'asl_alphabet_train', 'asl_alphabet_train')
        self.data = {l:None for l in letters}
    
    def change_test_names(self):
        os.chdir(os.path.join(self.cwd, self.test_dir))
        for l in letters:
            os.rename(f'{l}_test.jpg', f'{l}0.jpg')
        os.chdir('..')
    
    def make_dirs(self):
        for l in letters:
            os.makedirs(os.path.join(self.cwd, 'data', l), exist_ok=True)
    
    def make_train_dirs(self):
        for l in letters:
            os.makedirs(os.path.join(self.cwd, 'train', l), exist_ok=True)
            os.makedirs(os.path.join(self.cwd, 'val', l), exist_ok=True)
            os.makedirs(os.path.join(self.cwd, 'test', l), exist_ok=True)

    def delete_extra_dirs(self):
        try:
            shutil.rmtree(os.path.join(self.cwd, 'archive'))
        except OSError as e:
            print(f'Error: {e}')
        
    def delete_dirs(self):
        try:
            shutil.rmtree(os.path.join(self.cwd, 'train'))
            shutil.rmtree(os.path.join(self.cwd, 'val'))
            shutil.rmtree(os.path.join(self.cwd, 'test'))
            os.remove('data.json')
        except OSError as e:
            print(f'Error: {e}')
    
    def make_data_file(self):
        # Move test examples to data/class_name folder
        self.make_dirs()
        self.change_test_names()
        for l in letters:
            start = os.path.join(self.cwd, self.test_dir, f'{l}0.jpg')
            end = os.path.join(self.cwd, 'data', l, f'{l}0.jpg')
            shutil.move(start, end)
        
        # Move train exammples to data/class_name folder 
        train_start = os.path.join(self.cwd, self.train_dir)
        for l in letters:
            current_dir = os.path.join(train_start, l)
            jpg_files = [file for file in os.listdir(current_dir) if file.lower().endswith('.jpg')]
            for file in jpg_files:
                start = os.path.join(current_dir, file)
                end = os.path.join(self.cwd, 'data', l, file)
                shutil.move(start, end)

        self.delete_extra_dirs()
        self.ok = True


    def make_splits(self,data_ratio, make_data,train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, export=True):

        if make_data:
            self.make_data_file()


        for l in letters: 
            jpg_files = [file for file in os.listdir(os.path.join(self.cwd, 'data', l)) if file.lower().endswith('.jpg')]
            random.shuffle(jpg_files)
            num_points = int(data_ratio*len(jpg_files))
            self.data[l] = jpg_files[:num_points]


        jpg_files = []
        labels = []
        for l, files in self.data.items():
            jpg_files += files
            labels += [l]*num_points
        train_files, test_files, train_labels, test_labels = train_test_split(jpg_files, labels, test_size=1 - train_ratio, stratify=labels, random_state=42)
        val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels, test_size=test_ratio/(test_ratio + val_ratio), stratify=test_labels, random_state=42)
        
        
        self.make_train_dirs()
        for tr, lbl in zip(train_files, train_labels):
            start = os.path.join(self.cwd, 'data', lbl, tr)
            end = os.path.join(self.cwd, 'train', lbl)
            shutil.copy(start, end)
        
        for tr, lbl in zip(val_files, val_labels):
            start = os.path.join(self.cwd, 'data', lbl, tr)
            end = os.path.join(self.cwd, 'val', lbl)
            shutil.copy(start, end)

        for tr, lbl in zip(test_files, test_labels):
            start = os.path.join(self.cwd, 'data', lbl, tr)
            end = os.path.join(self.cwd, 'test', lbl)
            shutil.copy(start, end)

        if export:
            self.data.clear()
            self.data['train'] = train_files
            self.data['val'] = val_files
            self.data['test'] = test_files
            with open('data.json', 'w') as fp:
                json.dump(self.data, fp)

    def recreate_json(self):
        with open('data.json', 'r') as fp:
            self.data = json.load(fp)

        self.make_train_dirs()

        for split in self.data.keys():
            files = self.data[split]
            for f in files:
                start = os.path.join(self.cwd, 'data' ,f'{f[0]}', f)
                end = os.path.join(self.cwd, split, f'{f[0]}', f)
                shutil.copy(start, end)

dm = DataManager()
dm.make_splits(make_data=False, export=True, data_ratio=0.3)  

