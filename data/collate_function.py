# encoding: utf-8

def build_collate_fn():
    def collate_fn(batch):
        if len(batch[0]) == 2:
            data, batch_converters, = zip(*batch)
            batch_converter = batch_converters[0]
            data, anns = batch_converter(data)
            anns = None
        elif len(batch[0]) == 3:
            data, anns, batch_converters, = zip(*batch)
            batch_converter = batch_converters[0]
            data, anns = batch_converter(data, anns)
        else:
            raise Exception("Unexpected Num of Components in a Batch")

        return data, anns

    return collate_fn



