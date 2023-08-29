from torch.utils.data import IterableDataset

class HintDataSet(IterableDataset): 
    # def __init__(self, 
    #              start_index: int, 
    #              end_index: int, 
    #              read_data_func: function,
    #              input_categories: list,
    #              output_categories: list
    #              ):
    #     self.read_data_func = read_data_func
    #     self.input_categories = input_categories
    #     self.output_categories = output_categories
    #     self.start_index = start_index
    #     self.end_index = end_index

    # def __len__(self):
    #     return len(self.end_index - self.start_index + 1)

    # def __getitem__(self, idx):
    #     input_list = self.read_data_func(idx, self.input_categories)
    #     output_list = self.read_data_func(idx, self.output_categories)
    #     return [input_list, output_list]
    def __init__(self, 
                 start_index: int, 
                 end_index: int, 
                 read_text_iterator,
                 input_categories: list,
                 output_categories: list):
        self.start_index = start_index
        self.end_index = end_index
        src_data_iter = read_text_iterator(start_index, end_index, input_categories)
        trg_data_iter = read_text_iterator(start_index, end_index, output_categories)
        self._iterator = zip(src_data_iter, trg_data_iter)
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == (self.end_index - self.start_index + 1) - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.end_index - self.start_index + 1

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos

    def __str__(self):
        return 'hint data set' # self.description
    