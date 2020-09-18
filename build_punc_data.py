from utils.dataio import load_txt_data, load_file_name
from tqdm import trange, tqdm
import re
import logging
import random

PUNCS_NUM_LABEL = [0, 1, 2, 3]


class DataBuilder:
    def __init__(self, data_dir, random_seed=1024, max_num=200000):
        """

        :param data_dir:
        :param random_seed:
        :param max_num: maximum number of sentence in one file
        """
        random.seed(random_seed)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.data_dir = data_dir
        self.logger.info("  root = %s", self.data_dir)

        self.file_list = load_file_name(self.data_dir)[2]
        self.logger.info("  training books = %s", ' '.join(self.file_list))

        self.max_num = max_num
        self.raw_data = self.read_file(self.file_list)

    def read_file(self, file_names):
        """

        :param file_names:
        :return:

        load data: 按文本自然段读取
                [
                    ["para1"],
                    ["para2"],
                    ]
        """
        res = []
        count = 0
        for file_name in tqdm(file_names, desc='Load Data'):
            raw = load_txt_data(self.data_dir + file_name, origin=True)
            count += len(raw)
            for item in tqdm(raw, desc='adding data from {}'.format(file_name)):
                res.append(item)
        self.logger.info("  Num paragraph = %d", count)
        return res

    def select_data(self, data):

        for paragraph in data:
            paragraph = paragraph.strip()
            paragraph = self.format_puncs_space(paragraph)

    def format_puncs_space(self, sentence):
        for punc in self.puncs_pattern:
            _punc1 = ' ' + punc
            _punc2 = punc + ' '
            while _punc1 in sentence:
                sentence = re.sub(_punc1, punc, sentence)

            while _punc2 in sentence:
                sentence = re.sub(_punc2, punc, sentence)

        return sentence

if __name__ == '__main__':
    _data_builder = DataBuilder('data/segmentation_corpus/raw/')
