import io
import os
import codecs
import torch
import random
import numpy as np
import json
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import accuracy_score
from collections import defaultdict


def set_random_seed(random_seed):
    # This is the random_seed of hope.
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # check whether random.
    # torch.use_deterministic_algorithms(True)


def labels_to_multihot(labels, num_classes=146):
    multihot_labels = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        for l in label:
            multihot_labels[i][l] = 1
    return multihot_labels


def get_precision_recall_f1(y_true: np.array, y_pred: np.array, average='micro'):
    precision = metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(
        y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def get_precision_recall_f1_curve(y_true: np.array, y_pred: np.array):
    result = defaultdict(list)
    for threshold in np.linspace(0, 0.5, 51):
        result['threshold'].append(threshold)
        tmp_y_pred = np.array(y_pred >= threshold, np.int64)

        p, r, f1_micro = get_precision_recall_f1(y_true, tmp_y_pred, 'micro')
        result['precision_micro'].append(round(p, 4))
        result['recall_micro'].append(round(r, 4))
        result['f1_micro'].append(round(f1_micro, 4))

        p, r, f1_macro = get_precision_recall_f1(y_true, tmp_y_pred, 'macro')
        result['precision_macro'].append(round(p, 4))
        result['recall_macro'].append(round(r, 4))
        result['f1_macro'].append(round(f1_macro, 4))

        result['f1_avg'].append(round((f1_micro+f1_macro)/2, 4))
    return result


def evaluate(valid_dataloader, model, tokenizer, device, args, tokenizer_mode='char'):
    model.eval()
    all_predictions = []
    all_labels = []
    for i, data in enumerate(valid_dataloader):
        facts, labels = data
        
        # move data to device
        labels = torch.from_numpy(np.array(labels)).to(device)
        
        # tokenize the data text
        if tokenizer_mode == 'word':
            inputs, inputs_seq_lens = tokenizer.tokenize_seq(list(facts))
            inputs = inputs.to(device)
            # inputs = tokenizer.tokenize(list(facts)).to(device)
        elif tokenizer_mode == 'char':
            inputs = tokenizer(list(facts), max_length=args.input_max_length,
                            padding=True, truncation=True, return_tensors='pt')
        else:
            raise NameError

        with torch.no_grad():
            # forward
            logits = model(inputs, inputs_seq_lens)
        
        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1), 'macro')
    return accuracy, p_macro, r_macro, f1_macro


class WordTokenizer(object):
    def __init__(self, data_type, max_length=None):
        path_prefix = './datasets/'
        if data_type == 'CAIL2018':
            vocab_path = os.path.join(path_prefix, data_type , data_type + '_word2id.json')
            train_path = os.path.join(path_prefix, data_type , data_type + '_process_train.json')

        if not Path(vocab_path).exists():
            self.word2id = self.build_vocab(train_path, vocab_path)
        else:
            self.word2id = json.load(open(vocab_path))
        
        self.max_length = max_length
        print(f'Vocabulary size is: {len(self.word2id)}')

    def get_word2id(self):
        return self.word2id

    # TODO: set min_count via word frequency
    def build_vocab(self, train_path, word2id_path):
        word_index = 1 # index 1 for <UNK>, index 0 for <PAD>
        word2id = {'UNK': word_index}
        word_index += 1
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_dict = json.loads(line)
                fact_seg = json_dict['fact_seg']
                for word in fact_seg.split(' '):
                    if word not in word2id:
                        word2id[word] = word_index
                        word_index += 1
        print("save word2id.")
        fw_word2id = codecs.open(word2id_path, 'w', encoding='utf-8')
        json.dump(word2id, fw_word2id, ensure_ascii=False, indent=4)
        return word2id

    def vocab_size(self):
        return len(self.word2id) + 1

    def token2id(self, token):
        if not (token in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[token]

    def tokenize(self, texts):
        input_ids = torch.LongTensor(len(texts), self.max_length).zero_() # 0 for <PAD>
        
        for t_id, text in enumerate(texts):
            text = text.split(' ')
            for w_id, word in enumerate(text):
                if w_id >= self.max_length:
                    break
                input_ids[t_id][w_id] = self.token2id(word)
        return input_ids
    
    def tokenize_seq(self, texts):
        input_ids = torch.LongTensor(len(texts), self.max_length).zero_() # 0 for <PAD>
        input_seq_lens = []
        
        for t_id, text in enumerate(texts):
            text = text.split(' ')
            input_seq_lens.append(min(len(text), self.max_length))
            for w_id, word in enumerate(text):
                if w_id >= self.max_length:
                    break
                input_ids[t_id][w_id] = self.token2id(word)
        return input_ids, torch.tensor(input_seq_lens)
        

# load word embedding
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = tokens[1:]
    return data 


# get embedding matrix
def get_embedding_matrix(word_embedding, word2id, victor_size=None):
    # index 0 for <PAD>. The features of <PAD> are zeros.
    embedding_matrix = np.zeros((len(word2id) + 1, victor_size))
    count = 0
    for word, i in word2id.items():
        word_vector = word_embedding[word] if word in word_embedding else None
        if word_vector is not None:
            count = count + 1
            embedding_matrix[i] = np.array(word_vector, dtype=float)
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    print(embedding_matrix.shape, f'OOV ratio is: {1 - count * 1.0 / embedding_matrix.shape[0]}')
    return embedding_matrix


def sequence_mask(lengths, max_len=None):
    '''codes are from: https://blog.csdn.net/anshiquanshu/article/details/112433323'''
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size, max_len)
    .lt(lengths.unsqueeze(1)))


def dataset_name_to_charges(dataset_name) -> list:
    if dataset_name == "CAIL2018":
        result = []
        file_path = "datasets/CAIL2018/CAIL2018_label2index.txt"
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                result.append(line.split("\t")[1])
        return result
        # return ['故意伤害', '盗窃', '诈骗', '寻衅滋事', '危险驾驶', '抢劫', '虚开增值税专用发票、用于骗取出口退税、抵扣税款发票', '强奸', '交通肇事', '合同诈骗', '信用卡诈骗',
        #         '保险诈骗', '聚众斗殴', '贷款诈骗', '票据诈骗', '掩饰、隐瞒犯罪所得、犯罪所得收益', '过失致人死亡', '滥用职权', '非法采矿', '走私、贩卖、运输、制造毒品',
        #         '生产、销售有毒、有害食品', '故意杀人', '非法经营', '故意毁坏财物', '非法拘禁', '非法持有毒品', '抢夺', '妨害公务', '拒不支付劳动报酬', '敲诈勒索', '窝藏、包庇',
        #         '开设赌场', '非法持有、私藏枪支、弹药', '赌博', '非法种植毒品原植物', '骗取贷款、票据承兑、金融票证', '滥伐林木', '职务侵占', '非法占用农用地', '重大责任事故',
        #         '组织、领导传销活动', '生产、销售假药', '生产、销售伪劣产品', '贪污', '破坏广播电视设施、公用电信设施', '拒不执行判决、裁定', '失火', '生产、销售不符合安全标准的食品',
        #         '过失致人重伤', '放火', '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物', '挪用资金', '非法吸收公众存款', '非法猎捕、杀害珍贵、濒危野生动物', '盗掘古文化遗址、古墓葬',
        #         '盗伐林木', '非法收购、运输盗伐、滥伐的林木', '伪造、变造、买卖国家机关公文、证件、印章', '爆炸', '挪用公款', '冒充军人招摇撞骗', '非法行医', '非法进行节育手术',
        #         '拐卖妇女、儿童', '虚开发票', '侵犯著作权', '单位行贿', '过失以危险方法危害公共安全', '非法获取公民个人信息', '投放危险物质', '扰乱无线电通讯管理秩序', '破坏生产经营',
        #         '玩忽职守', '非法侵入住宅', '聚众扰乱社会秩序', '招摇撞骗', '伪证', '以危险方法危害公共安全', '销售假冒注册商标的商品', '违法发放贷款', '非法转让、倒卖土地使用权',
        #         '重大劳动安全事故', '非法捕捞水产品', '拐骗儿童', '持有伪造的发票', '非法处置查封、扣押、冻结的财产', '聚众扰乱公共场所秩序、交通秩序', '编造、故意传播虚假恐怖信息',
        #         '非法出售发票', '对非国家工作人员行贿', '诬告陷害', '假冒注册商标', '重婚', '出售、购买、运输假币', '非法买卖制毒物品', '集资诈骗', '走私普通货物、物品',
        #         '持有、使用假币', '破坏易燃易爆设备', '伪造、变造金融票证', '妨害信用卡管理', '破坏电力设备', '绑架', '污染环境', '伪造公司、企业、事业单位、人民团体印章', '串通投标',
        #         '介绍贿赂', '遗弃', '制作、复制、出版、贩卖、传播淫秽物品牟利', '伪造、变造居民身份证', '非国家工作人员受贿', '非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品',
        #         '帮助犯罪分子逃避处罚', '强制猥亵、侮辱妇女', '非法采伐、毁坏国家重点保护植物', '猥亵儿童', '非法狩猎', '行贿', '强迫交易']
    else:
        raise ValueError("Invalid dataset name")


if __name__ == '__main__':
    # word_tokenizer = WordTokenizer(data_type='criminal_s', max_length=10)
    # texts = ['我 爱 自然 语言 处理', '我 爱 自然 语言 处理']
    # input_ids = word_tokenizer.tokenize(texts)
    # print(input_ids)
    a = dataset_name_to_charges("CAIL2018")
    print(a)