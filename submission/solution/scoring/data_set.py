from .read_solution import get_dict_from_solution


class DataSet:

    def __init__(self, items=None, file_name=None):
        self.items = get_dict_from_solution(file_name) if file_name is not None else items if items is not None else []
        self.items_len = len(self.items)
        self.items_pos = 0

    def get_item_from_threshold(self, frame_number, threshold, remember_pos=False):
        left, right = self.get_left_right_attribute(frame_number, remember_pos=remember_pos)
        if left is not None and frame_number - left['frame'] <= threshold:
            return left
        elif right is not None and right['frame'] - frame_number <= threshold:
            self.items_pos += 1
            return right
        else:
            return None

    def get_left_right_attribute(self, frame_number, remember_pos=False):
        left = None
        for right in self.items[self.items_pos if remember_pos else 0:]:
            if frame_number < int(right['frame']):
                return left, right
            left = right
            self.items_pos += 1 if remember_pos else 0
        # self.items_pos = self.items_pos % len(self.items)
        return left, None

    def __iter__(self):
        for i in self.items:
            yield i

    def __len__(self):
        return self.items_len

    def num_excess(self):
        return len(self.items) - self.items_len

    def add_item(self, item):
        self.items_len += 1 if item is not None else 0
        self.items.append(item)
