import re
import logging
from tqdm import trange, tqdm
import random
import pyparsing
from utils.dataio import load_txt_data, save_txt_file, load_file_name


class SegmentationData(object):
    def __init__(self, raw_path, punc_conf_path, seg_conf_path, seed=12):

        random.seed(seed)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.dir = raw_path
        self.logger.info("  root = %s", self.dir)

        self.book_list = load_file_name(self.dir)[2]
        self.logger.info("  training books = %s", ' '.join(self.book_list))

        self.sentence_end_symbol = list("。！？?!")

        self.max_sentence_in_paragraph = 8
        self.max_char_in_sentence = 64
        self.max_char_in_paragraph = 128
        self.max_char_in_batch = 512

        # self.punc_labels = [x.split('\t')[0] for x in load_txt_data(punc_conf_path)]
        self.puncs = [x.split('\t')[1] for x in load_txt_data(punc_conf_path)]
        self.puncs_pattern = ['\\' + x.split('\t')[1] for x in load_txt_data(punc_conf_path)]
        # self.punc_label_dict = dict(zip(self.puncs, self.punc_labels))
        # self.label_punc_dict = dict(zip(self.punc_labels, self.puncs))

        self.segment_labels = [x.split('\t')[0] for x in load_txt_data(seg_conf_path)]
        self.segment_symbol = [x.split('\t')[1] for x in load_txt_data(seg_conf_path)]
        self.segment_label_symbol_dict = dict(zip(self.segment_labels, self.segment_symbol))
        # self.segment_symbol_label_dict = dict(zip(self.segment_symbol, self.segment_labels))

        self.place_holder = 'b'

        self.raw_data = []
        self.read_file(self.book_list)
        self.raw_data = self.format_data()
        self.punc_data = self.tag_punc_data()
        # self.seg_data = self.tag_seg_data()

    def format_puncs_space(self, sentence):
        for punc in self.puncs_pattern:
            _punc1 = ' ' + punc
            _punc2 = punc + ' '
            while _punc1 in sentence:
                sentence = re.sub(_punc1, punc, sentence)

            while _punc2 in sentence:
                sentence = re.sub(_punc2, punc, sentence)

        return sentence

    def split_paragraph(self, paragraph):
        """

        :param paragraph: str
        :return:
        :rtype: list: ['s1', 's2', 's3',]
        """
        i = 0
        j = 0
        new_paragraph = []
        while i < len(paragraph):
            if paragraph[i] in self.sentence_end_symbol:
                new_paragraph.append(paragraph[j:i + 1])
                j = i + 1
            i += 1
        if not new_paragraph:
            new_paragraph = [paragraph]

        return new_paragraph

    def remove_none_puncs_paragraph(self, paragraph):
        """
        :param paragraph: str
        :return: str
        """
        new_paragraph = None
        for punc in self.puncs:

            if punc in paragraph:
                new_paragraph = paragraph
                break

        return new_paragraph

    def read_file(self, file_names):
        """
        :param file_names:
        :return:

        load data: [
                    ["sentence1", "sentence2", "sentence3"],
                    ["sentence1", "sentence2", "sentence3"],
                    ]
        """

        for file_name in file_names:
            raw = load_txt_data(self.dir + file_name, origin=True)[:200000]
            for i in trange(len(raw), desc='Load {}'.format(file_name)):
                paragraph: str = raw[i].strip()
                paragraph: str = self.remove_none_puncs_paragraph(paragraph)
                if paragraph:
                    paragraph: str = re.sub('\\u3000', ':', paragraph)
                    paragraph: str = self.format_puncs_space(paragraph)
                    paragraph: list = self.split_paragraph(paragraph)
                    self.raw_data.append(paragraph)

        self.logger.info("  Num paragraph = %d", len(self.raw_data))

    def format_data(self):
        data = []
        max_len = 0
        for i in trange(len(self.raw_data), desc='Format data'):
            paragraph = self.raw_data[i]

            if len(paragraph) > self.max_sentence_in_paragraph:
                continue
            flag = True
            for j in range(len(paragraph)):
                if len(paragraph[j]) > self.max_char_in_sentence:
                    flag = False
            if not flag:
                continue
            max_len = max(max_len, len(''.join(paragraph)))
            data.append(paragraph)
        print(max_len)
        return data

    def tag_punc_data(self):

        punc_data, punc_label = [], []

        for i in trange(len(self.raw_data), desc='tagging punc data'):
            section = self.place_holder + ''.join(self.raw_data[i])
            j = 0
            sub_punc_data, sub_punc_label = [], []
            while j < len(section):
                if section[j] in self.puncs:
                    sub_punc_label[-1] = section[j]
                else:
                    sub_punc_data.append(section[j])
                    sub_punc_label.append('word')
                j += 1
            punc_data.append(sub_punc_data)
            punc_label.append(sub_punc_label)

        res = []

        for i in trange(len(punc_data), desc='format tagged data'):
            for j in range(len(punc_data[i])):
                if j == len(punc_data[i]) - 1:
                    res.append('{} {}\n'.format(punc_data[i][j], punc_label[i][j]))
                else:
                    res.append('{} {}'.format(punc_data[i][j], punc_label[i][j]))
        return res

    @staticmethod
    def cut_sentence(sentence, max_cut):
        cut = random.randint(1, max_cut)
        return sentence[0:cut]

    def tag_seg_paragraph(self, paragraph, cut=False, finish=False):
        res = []
        for j in range(len(paragraph)):
            char = paragraph[j]
            if j == 0:
                label = self.segment_label_symbol_dict['paragraph_b']
            elif j == len(paragraph) - 1:  # Last One
                if cut:
                    label = self.segment_label_symbol_dict['cut']
                else:
                    label = self.segment_label_symbol_dict['paragraph_e']
            else:
                label = self.segment_label_symbol_dict['word']
            res.append('{} {}\n'.format(char, label))

        if finish:
            res.append('\n')
        return res

    def tag_seg_data(self):

        raw = []
        for i in trange(len(self.raw_data), desc='preprocess data'):
            paragraph = ''.join(self.raw_data[i])
            for item in self.puncs_pattern:
                paragraph = re.sub(item, '', paragraph)

            if len(paragraph) <= 3:
                continue
            raw.append(paragraph)
        print(len(raw))
        res = []
        para_num = random.randint(1, 10)
        k = 0  # k paragraph in batch need reset
        doc_len = 0  # record doc len

        batch_num = 0

        i = 0  # paragraph index not need reset
        from time import time
        start = time()
        while i < len(raw):
            paragraph = raw[i]
            doc_len += len(paragraph)
            # print(doc_len)

            if doc_len > self.max_char_in_batch:
                para_num = k
                max_cut = len(paragraph) - (doc_len - self.max_char_in_batch)

                paragraph = self.cut_sentence(paragraph, max_cut)

            elif doc_len == self.max_char_in_batch:
                para_num = k

            if k < para_num:
                _ = self.tag_seg_paragraph(paragraph, cut=False)
                res.append(_)
                k += 1

            elif k == para_num:
                if doc_len > self.max_char_in_batch:
                    # TODO: 进入这里证明已经被剪切，最后一个label为 cut
                    _ = self.tag_seg_paragraph(paragraph, cut=True, finish=True)
                    res.append(_)

                else:
                    cut = random.randint(1, 10)
                    if len(paragraph) > 4:
                        if cut <= 3:
                            max_cut = len(paragraph) - 2
                            paragraph = self.cut_sentence(paragraph, max_cut)
                            _ = self.tag_seg_paragraph(paragraph, cut=True, finish=True)
                            res.append(_)
                        else:
                            _ = self.tag_seg_paragraph(paragraph, cut=False, finish=True)
                            res.append(_)
                    else:
                        _ = self.tag_seg_paragraph(paragraph, cut=False, finish=True)
                        res.append(_)

                # TODO: reset para
                k = 0
                doc_len = 0
                para_num = random.randint(1, 10)
                batch_num += 1

            if i % 2000 == 0:
                self.logger.info("  tagging seg data %d / {}".format(len(raw)), i)
            i += 1
        _res = []
        for item in tqdm(res):

            for jtem in item:
                _res.append(jtem)
        print(batch_num)
        return _res


if __name__ == '__main__':
    _a = SegmentationData('../data/segmentation_corpus/raw/', '../utils/config/punctuation.dat',
                          '../utils/config/segmentation.dat')
    #
    # for item in _a.raw_data:
    #     # print([item])
    #     pass
    # print(len(_a.raw_data))
    # print(_a.max_seq)
    save_txt_file(_a.punc_data, '../data/segmentation_corpus/punc_data.train.txt')
    # save_txt_file(_a.seg_data, '../data/segmentation_corpus/segment_data.train.txt', end='')
