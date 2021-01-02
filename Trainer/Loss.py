from torch import nn
import torch
import math

class CrossEntropyLossWithGaussianSmoothedLabels(nn.Module):
    def __init__(self, num_classes = 722, blur_range = 3):
        super(CrossEntropyLossWithGaussianSmoothedLabels,self).__init__()
        self.dim = -1
        self.num_classes = num_classes
        self.blur_range = blur_range
        self.gaussian_decays = [self.gaussian_val(dist=d) for d in range(blur_range + 1)]
        print("debug")
    
    def gaussian_val(self, dist,sigma=1):
        return math.exp(-math.pow(2, dist) / (2 * math.pow(2, sigma)))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        #predinction: (batch, 31, 722)
        #target: (b,31)
        prediction_softmax = torch.log_softmax(prediction,dim=self.dim)
        target_smoothed = self.smoothed_label(target)
        target_loss_sum = -(prediction_softmax * target_smoothed).sum(dim=self.dim)
        return target_loss_sum.mean()

    def smoothed_label(self, target: torch.Tensor):
        target_onehot = self.empty_onehot(target,self.num_classes).to(target.device)
        target_smoothed = self.gaussian_blur(target, target_onehot)
        target_smoothed = self.to_onehot(target, self.num_classes, target_smoothed)
        return target_smoothed
    
    def empty_onehot(self, target: torch.Tensor, num_classes):
        onehot_size = target.size() + (num_classes,)
        return torch.FloatTensor(*onehot_size).zero_()
    
    def to_onehot(self,target: torch.Tensor, num_classes: int, src_onehot: torch.Tensor = None):
        if src_onehot is None:
            one_hot = self.empty_onehot(target, num_classes)
        else:
            one_hot = src_onehot
        last_dim = len(one_hot.size()) - 1
        with torch.no_grad():
            one_hot = one_hot.scatter_(
                dim=last_dim, index=torch.unsqueeze(target, dim=last_dim), value=1.0)
        return one_hot
    
    def gaussian_blur(self, target: torch.Tensor, one_hot: torch.Tensor):
        with torch.no_grad():
            for dist in range(self.blur_range,-1,-1):
                one_hot = self.set_decayed_values(dist, target, one_hot)
        return one_hot
    
    def set_decayed_values(self, dist,  target_idx: torch.Tensor, one_hot: torch.Tensor):
        for direction in [1,-1]:
            blur_idx = torch.clamp(
                target_idx + (direction * dist),min=0,max=self.num_classes-1
            )
            decayed_val = self.gaussian_decays[dist]
            one_hot = one_hot.scatter_(
                dim=2, index=torch.unsqueeze(blur_idx, dim=2), value=decayed_val)
        return one_hot

    
if __name__ == '__main__':
    loss = CrossEntropyLossWithGaussianSmoothedLabels(num_classes=7)
    k = torch.tensor([[4,2,0,4]])
    print(k)
    print(loss.smoothed_label(k))