from torch import nn

class Thresholder(nn.Module):
    def __init__(self, device='cpu', op='gt', th=0):
        super().__init__()
        self.op = op
        self.th = th

    def forward(self, src_img, **kwargs):
        op = self.op
        th = self.th
        if op == 'gt':
            result = src_img > th
        elif op == 'lt':
            result = src_img < th
        elif op == 'eq':
            result = src_img == th
        elif op == 'bw':
            result = (src_img > th[0]) * (src_img < th[1])
        else:
            raise Exception(f"Unsupported thresholding op: '{op}'")
        return result


def create(**kwargs):
    return Thresholder(**kwargs)
