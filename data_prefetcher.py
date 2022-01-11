import torch


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            (self.img, self.class_map, self.regression_map) = next(self.loader)
        except StopIteration:
            self.img = None
            self.class_map = None
            self.regression_map = None
            return
        with torch.cuda.stream(self.stream):
            self.img = self.img.cuda(non_blocking=True)
            self.class_map = self.class_map.cuda(non_blocking=True)
            self.regression_map = self.regression_map.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.img
        class_map = self.class_map
        regression_map = self.regression_map
        self.preload()
        return img, class_map, regression_map

