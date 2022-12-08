'''
nohup python -u train.py --model_type=TextCNN --gpu_id=1 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_CNN.pth > logs/CAIL2018_TextCNN.log &
nohup python -u train.py --model_type=TextRNN --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_RNN.pth > logs/CAIL2018_TextRNN.log &
nohup python -u train.py --model_type=Transformer --gpu_id=3 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_Transformer.pth > logs/CAIL2018_Transformer.log &

nohup python -u train.py --model_type=LSTM --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_LSTM.pth -log=logs/CAIL2018_LSTM.log &
'''

from sklearn import metrics
from sklearn.metrics import accuracy_score

from utils import set_random_seed
from model import LawFormer
from dataset import WordCaseData
import argparse
import os
import logging

import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np

# pd.set_option('display.max_columns', None)
def evaluate(valid_dataloader, model, device):
    model.eval()
    all_predictions = []
    all_labels = []
    for i, data in enumerate(valid_dataloader):
        facts, labels = data

        # move data to device
        labels = torch.from_numpy(np.array(labels)).to(device)

        with torch.no_grad():
            # forward
            logits = model(facts)

        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1),
                                                                   'macro')
    return accuracy, p_macro, r_macro, f1_macro

def get_precision_recall_f1(y_true: np.array, y_pred: np.array, average='micro'):
    precision = metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(
        y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Charge Prediction")
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='default: 128')
    parser.add_argument('--epochs', type=int, default=30, help='default: 30')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='default: 1e-3')
    parser.add_argument('--input_max_length', '-l', type=int, default=500, help='default: 500')
    parser.add_argument('--random_seed', type=int, default=3407, help='default: 3407')
    parser.add_argument('--num_classes', type=int, default=119, help='default: 119')
    parser.add_argument('--model_type', type=str, default='LSTM',
                        help='[TextRNN, TextCNN, TextRCNN, TextAttRNN, Transformer, SAttCaps, Capsule]')
    parser.add_argument('--gpu_id', type=str, default='0', help='default: 0')
    parser.add_argument('--resume', '-r', action='store_true', help='default: False')
    parser.add_argument('--word_embed_path', type=str, default='./datasets/word_embed/small_w2v.txt')
    parser.add_argument('--dataset_type', type=str, default='CAIL2018', help='[CAIL2018]')
    parser.add_argument('--save_path', '-s', type=str, default='./checkpoints/model_baseline_best.pth')
    parser.add_argument('--log_file_name', '-log', type=str, default='./logs/model_baseline_best.log')
    parser.add_argument('--resume_checkpoint_path', '-c', type=str, default='./checkpoints/model_baseline_best.pth')
    args = parser.parse_args()

    args.model_type = 'LawFormer'
    args.log_file_name = 'logs/CAIL2018/logs/{}.log'.format(args.model_type)
    args.save_path = 'logs/CAIL2018/checkpoints/{}.pth'.format(args.model_type)
    args.resume_checkpoint_path = args.save_path
    args.gpu_id = '0'
    args.num_classes = 119

    logging.basicConfig(filename=args.log_file_name,
                        level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.info(args)

    # check the device
    device = 'cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))

    torch.cuda.empty_cache()

    # seed random seed
    set_random_seed(args.random_seed)

    # prepare training data
    train_path = 'datasets/CAIL2018/CAIL2018_train.json'
    valid_path = 'datasets/CAIL2018/CAIL2018_valid.json'
    test_path = 'datasets/CAIL2018/CAIL2018_test.json'

    logging.info(f'Train_path: {train_path}')
    logging.info(f'Valid_path: {valid_path}')
    logging.info(f'Test_path: {test_path}')

    training_data = WordCaseData(mode='train', train_file=train_path)
    valid_data = WordCaseData(mode='valid', valid_file=valid_path)
    test_data = WordCaseData(mode='test', test_file=test_path)

    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=training_data.collate_function, drop_last=True)
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate_function)
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate_function)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load the model
    if args.model_type == 'LawFormer':
        model = LawFormer(device, num_classes=args.num_classes)
    else:
        raise NameError

    logging.info(f'Load {args.model_type} model.')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # resume checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 'cpu' to 'gpu'
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logging.info(
            f"Resume model and optimizer from checkpoint '{args.resume_checkpoint_path}' with epoch {checkpoint['epoch']} and best F1 score of {checkpoint['best_f1_score']}")
        logging.info(f"optimizer lr: {optimizer.param_groups[0]['lr']}")
        start_epoch = checkpoint['epoch']
        best_f1_score = checkpoint['best_f1_score']
    else:
        # start training process
        start_epoch = 0
        best_f1_score = 0

    model.to(device)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for i, data in enumerate(train_dataloader):
            facts, labels = data

            # forward and backward propagations
            loss, logits = model(facts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                predictions = logits.softmax(
                    dim=1).detach().cpu().numpy()
                labels = labels.cpu().numpy()

                logging.info(
                    f'epoch{epoch + 1}, step{i + 1:5d}, loss: {loss.item():.4f}')

                pred = np.argmax(predictions, axis=1)
                accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(
                    labels, pred, 'macro')
                logging.info(
                    f'train accusation macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')

        if (epoch + 1) % 1 == 0:
            logging.info('Evaluating the model on validation set...')
            accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = evaluate(valid_dataloader, model,device)
            logging.info(
                f'valid accusation macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')
            # scheduler.step(f1_macro_accu)

            if f1_macro_accu > best_f1_score:
                best_f1_score = f1_macro_accu
                logging.info(
                    f"the valid best average F1 score is {best_f1_score}.")
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_f1_score': best_f1_score,
                }
                torch.save(state, args.save_path)
                logging.info(f'Save model in path: {args.save_path}')

    # Load Best Checkpoint
    logging.info('Load best checkpoint for testing model.')
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = evaluate(test_dataloader, model, device)
    logging.info(
        f'test accusation macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')
