import argparse
import os
from collections import defaultdict, Counter
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="../data/penn-tree-bank/raw/wsj", type=str, required=True,
        help="The input data directory."
    )
    parser.add_argument(
        "--train_sections", default="train", type=str, required=True,
        help="The PTB sections to use: 'train', 'test', 'dev', or 'all'."
    )
    parser.add_argument(
        "--test_sections", default="test", type=str, required=True,
        help="The PTB sections to use: 'train', 'test', 'dev', or 'all'."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output data directory."
    )
    parser.add_argument(
        "--cutoff", default=10, type=int, required=True,
        help="The sentence position frequency cut-off. Skip sentence positions with frequency lower than this value."
    )
    parser.add_argument(
        "--n", default=3, type=int, required=True,
        help="The order of the n-gram language model."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for initialization."
    )
    args = parser.parse_args()

    dev_sections = ["01"]
    train_sections = ["{:02d}".format(sec) for sec in range(0, 21)]
    test_sections = ["{:02d}".format(sec) for sec in range(21, 25)]
    all_sections = ["{:02d}".format(sec) for sec in range(0, 25)]

    if args.train_sections == 'train':
        training_sections = train_sections
    elif args.train_sections == 'test':
        training_sections = test_sections
    elif args.train_sections == 'dev':
        training_sections = dev_sections
    elif args.train_sections == 'all':
        training_sections = all_sections
    else:
        raise ValueError("Invalid training sections identifier '%s'" % args.train_sections)

    if args.test_sections == 'train':
        testing_sections = train_sections
    elif args.test_sections == 'test':
        testing_sections = test_sections
    elif args.test_sections == 'dev':
        testing_sections = dev_sections
    elif args.test_sections == 'all':
        testing_sections = all_sections
    else:
        raise ValueError("Invalid training sections identifier '%s'" % args.test_sections)

    sentence_positions = []
    sentences = []
    for sec in testing_sections:
        for root, _, files in os.walk(os.path.join(args.data_path, sec)):
            for file in files:
                with open(os.path.join(root, file), 'r', encoding="ISO-8859-1") as f:
                    s_position = 0
                    for line in f:
                        line = line.strip('\n')
                        line = line.strip()
                        if line == '.START':
                            continue
                        if not line:
                            continue
                        s_position += 1
                        sentences.append(line)
                        sentence_positions.append(s_position)

    print("Number of sentences:", len(sentences))
    print("Max sentence position: ", max(sentence_positions))

    # Check frequency of each sentence position
    pos_counter = Counter(sentence_positions)
    print('Positions and their frequency:')
    print([(p, pos_counter[p]) for p in sorted(list(pos_counter.keys()))])

    # Check the highest sentence position for each frequency value
    highest_pos_by_freq = defaultdict(int)
    for turn, fr in pos_counter.items():
        if turn > highest_pos_by_freq[fr]:
            for _fr in range(1, fr + 1):
                highest_pos_by_freq[_fr] = turn

    # Only consider poisitions with at least k items in the test set (k=10 in Keller 2004)
    print('Frequency cut-off:', args.cutoff,
          '  Lowest position with this frequency:', highest_pos_by_freq[args.cutoff])

    tmp_sentences, tmp_positions = [], []
    for sentence, position in zip(sentences, sentence_positions):
        if pos_counter[position] >= args.cutoff:
            tmp_sentences.append(sentence)
            tmp_positions.append(position)

    tmp_dataset = list(zip(tmp_sentences, tmp_positions))
    tmp_lengths = [len(s.split()) for s in tmp_sentences]
    tmp_dataset = [x for (_, x) in sorted(zip(tmp_lengths, tmp_dataset), reverse=True)]

    sentences = [x[0] for x in tmp_dataset]
    positions = [x[1] for x in tmp_dataset]



    # df = pd.DataFrame({
    #     'position': results['position'],
    #     'h': results['entropy'],
    #     'normalised_h': results['normalised_entropy'],
    #     'length': results['length']
    # })
    # out_file_name = os.path.join(args.out_path, 'ptb_{}_{}_{}_{}_{}'.format(
    #     args.sections,
    #     'bi' if args.bidirectional else 'uni',
    #     args.cutoff,
    #     args.right_context,
    #     args.max_seq_len
    # ))
    # df.to_csv(
    #     '{}.zip'.format(out_file_name),
    #     index=False,
    #     compression=dict(
    #         method='zip', archive_name='{}.csv'.format(out_file_name))
    # )
