from .data_set import DataSet


def calculate_correct(correct, submitted):
    num_correct = 0
    num_attributes = 0
    for k, v in correct.items():
        if k != 'frame':
            num_attributes += 1
            if k in submitted and v == submitted[k]:
                num_correct += 1
    return 0 if num_attributes == 0 else num_correct / num_attributes


class Compare:

    def __init__(self, correct: DataSet, submitted: DataSet, threshold):
        self.correct = correct
        self.submitted = submitted
        self.threshold = threshold
        self.same_points = DataSet()
        self.compare()

    def compare(self):
        self.correct.items_pos = 0
        for i in self.submitted:
            item = self.correct.get_item_from_threshold(i['frame'], self.threshold, remember_pos=True)
            self.same_points.add_item(item)

    def num_correct(self):
        num_correct = 0
        for c, s in zip(self.same_points, self.submitted):
            if None not in [c, s]:
                num_correct += calculate_correct(c, s)
        return num_correct

    def description_score(self):
        num_correct = self.num_correct()
        return {
            'sum_frame_score': num_correct,
            'TP_frames': len(self.same_points),
            'FP_frames': len(self.submitted) - len(self.same_points),
            'FN_frames': len(self.correct) - len(self.same_points)
        }

    @property
    def score(self):
        score = self.description_score()
        sum_frame_score = score['sum_frame_score']
        TP = score['TP_frames']
        FP = score['FP_frames']
        FN = score['FN_frames']
        print(sum_frame_score,TP,FP,FN)
        return sum_frame_score / (TP + 0.5 * (FP + FN))

    def __str__(self):
        return 'Score: {:.2}\n'.format(self.score)
