from torch.utils.data import Dataset
import json

class WordCaseData(Dataset):
    def __init__(self, mode='train', train_file=None, valid_file=None, test_file=None):
        assert mode in ['train', 'valid', 'test'], f"mode should be set to the one of ['train', 'valid', 'test']"
        self.mode = mode
        self.dataset = []
        if mode == 'train':
            self.dataset = self._load_data(train_file)
            print(f'Number of training dataset: {len(self.dataset)}')
        elif mode == 'valid':
            self.dataset = self._load_data(valid_file)
            print(f'Number of validation dataset: {len(self.dataset)}')
        else:
            self.dataset = self._load_data(test_file)
            print(f'Number of test dataset: {len(self.dataset)}.')

    def __getitem__(self, idx):
        features_content = self.dataset[idx]['fact_seg']
        if self.mode in ['train', 'valid', 'test']:
            labels = self.dataset[idx]['accu_index']

            return features_content, labels
        else:
            raise NameError

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(batch):
        features_content, labels = zip(*batch)
        return features_content, labels
    
    def _load_data(self, file_name):
        dataset = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                json_dict = json.loads(line)
                dataset.append(json_dict)
        return dataset


class CaseDataLawFormer(Dataset):
    def __init__(self, mode='train', train_file=None, valid_file=None, test_file=None):
        assert mode in ['train', 'valid', 'test'], f"mode should be set to the one of ['train', 'valid', 'test']"
        self.mode = mode
        self.dataset = []
        if mode == 'train':
            self.dataset = self._load_data(train_file)
            print(f'Number of training dataset: {len(self.dataset)}')
        elif mode == 'valid':
            self.dataset = self._load_data(valid_file)
            print(f'Number of validation dataset: {len(self.dataset)}')
        else:
            self.dataset = self._load_data(test_file)
            print(f'Number of test dataset: {len(self.dataset)}.')

    def __getitem__(self, idx):
        features_content = self.dataset[idx]['fact']
        if self.mode in ['train', 'valid', 'test']:
            labels = self.dataset[idx]['accu_index']

            return features_content, labels
        else:
            raise NameError

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(batch):
        features_content, labels = zip(*batch)
        return features_content, labels

    def _load_data(self, file_name):
        dataset = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                json_dict = json.loads(line)
                dataset.append(json_dict)
        return dataset

if __name__ == '__main__':
    train_file = './datasets/CAIL2018/CAIL2018_process_train.json'
    valid_file = './datasets/CAIL2018/CAIL2018_process_valid.json'
    test_file = './datasets/CAIL2018/CAIL2018_process_test.json'
    training_data = WordCaseData(mode='train', train_file=train_file)
    print(len(training_data))
