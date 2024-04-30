from collections import Counter

from sklearn.metrics import classification_report

from plot import *
from utils import flatten_lists


class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags, remove_O=False):
        self.classes = ['O', 'B-PAR', 'I-PAR', 'B-MAT', 'I-MAT', 'B-TYPE', 'I-TYPE']
        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

    def report_scores(self, model_name):
        print("==={}分类器报告===\n".format(model_name),
              classification_report(self.golden_tags, self.predict_tags))

    def _remove_Otags(self):
        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self, title, name):
        """计算混淆矩阵"""
        print("\nConfusion Matrix:")
        plot_my_confusionMatrix(self.golden_tags, self.predict_tags, list(self.tagset), title, name)
