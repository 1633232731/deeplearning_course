'''
nohup python -u train.py --model_type=TextCNN --gpu_id=1 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_CNN.pth > logs/CAIL2018_TextCNN.log &
nohup python -u train.py --model_type=TextRNN --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_RNN.pth > logs/CAIL2018_TextRNN.log &
nohup python -u train.py --model_type=Transformer --gpu_id=3 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_Transformer.pth > logs/CAIL2018_Transformer.log &

nohup python -u train.py --model_type=LSTM --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_LSTM.pth -log=logs/CAIL2018_LSTM.log &



'''

from tokenize import Name
from utils import get_precision_recall_f1, evaluate, WordTokenizer, load_vectors, get_embedding_matrix, set_random_seed, \
    sequence_mask
from model import Transformer, LSTM
from dataset import WordCaseData
import argparse
import os
import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer
# from torch.optim import AdamW
import numpy as np
import pandas as pd

# pd.set_option('display.max_columns', None)

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

    args.model_type = 'LSTM'
    args.log_file_name = 'logs/CAIL2018/logs/{}.log'.format(args.model_type)
    args.save_path = 'logs/CAIL2018/checkpoints/{}.pth'.format(args.model_type)
    args.resume_checkpoint_path = args.save_path
    args.gpu_id = '0'
    args.num_classes = 149

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

    # load the tokenizer
    tokenizer = WordTokenizer(data_type=args.dataset_type, max_length=args.input_max_length)
    word2id = tokenizer.get_word2id()

    logging.info('Load word embedding...')
    word_embedding = load_vectors(fname=args.word_embed_path)

    embedding_matrix = get_embedding_matrix(word_embedding=word_embedding, word2id=word2id, victor_size=200)

    # load the model
    if args.model_type == 'LSTM':
        model = LSTM(device, embeddings_matrix=torch.from_numpy(embedding_matrix), num_classes=args.num_classes)

    elif args.model_type == 'Transformer':
        model = Transformer(device, embeddings_matrix=torch.from_numpy(embedding_matrix), num_classes=args.num_classes)
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

            # tokenize the data text
            inputs_id, inputs_seq_lens = tokenizer.tokenize_seq(list(facts))
            inputs_id = inputs_id.to(device)

            # move data to device
            labels = torch.from_numpy(np.array(labels)).to(device)

            # forward and backward propagations
            loss, logits = model(inputs_id, inputs_seq_lens, labels)

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
            accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = evaluate(valid_dataloader, model, tokenizer,
                                                                                device, args, tokenizer_mode='word')
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
    accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = evaluate(test_dataloader, model, tokenizer, device, args,
                                                                        tokenizer_mode='word')
    logging.info(
        f'test accusation macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')
