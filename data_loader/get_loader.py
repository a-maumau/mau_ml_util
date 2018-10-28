import torch
import torchvision.transforms as transforms

# mixing up
class Mixup(object):
    def __init__(self, alpha=0.1, prepro_transform=None):
        self.prepro_transform = prepro_transform
        self.alpha = alpha

    def __call__(self, data):
        # imgs, masks are tuple.
        imgs, labels = zip(*data)
        imgs = torch.stack(imgs, dim=0)
        labels = torch.stack(labels, dim=0)

        batch_size = imgs.shape[0]
        perm_index = torch.randperm(batch_size)

        if self.alpha > 0.:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.

        mixed_imgs = lam*imgs + (1-lam)*imgs[perm_index]
        mix_labels = labels[perm_index]

        if self.prepro_transform is not None:
            for i in range(batch_size):
                img = transforms.functional.to_pil_image(mixed_imgs[i], "L")
                img = self.prepro_transform(img)
                mixed_imgs[i] = img
        else:
            for i in range(batch_size):
                img = transforms.functional.to_pil_image(mixed_imgs[i], "L")
                mixed_imgs[i] = transforms.Normalize((.5,.5,.5),(.5,.5,.5))(img).unsqueeze(0)

        # we need lambda for backpropagating
        return mixed_imgs, labels, mix_labels, lam

# mixing up for segmentation
class SegmentationMixup(object):
    def __init__(self, alpha=0.1, prepro_transform=None):
        self.prepro_transform = prepro_transform
        self.alpha = alpha

    def __call__(self, data):
        # imgs, masks are tuple.
        imgs, masks = zip(*data)
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)

        batch_size = imgs.shape[0]
        perm_index = torch.randperm(batch_size)

        if self.alpha > 0.:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.

        mixed_imgs = lam*imgs + (1-lam)*imgs[perm_index]
        mix_masks = masks[perm_index, :]

        if self.prepro_transform is not None:
            for i in range(batch_size):
                img = transforms.ToPILImage(mode="RGB")(mixed_imgs[i])
                img = self.prepro_transform(img).unsqueeze(0)
                mixed_imgs[i] = img
        else:
            for i in range(batch_size):
                img = transforms.ToTensor()(mixed_imgs[i])
                mixed_imgs[i] = transforms.Normalize((.5,.5,.5),(.5,.5,.5))(img).unsqueeze(0)

        mix_masks = masks[perm_index]
        
        # we need lambda for backpropagating
        return mixed_imgs, masks, mix_masks, lam

    # mixup on normalized image
    """
    def old__call__(self, data):
        imgs, masks = zip(*data)
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)

        batch_size = imgs.shape[0]
        # every lambda will be sample from same parameter

        if self.alpha > 0.:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.
        perm_index = torch.randperm(batch_size)

        mixed_imgs = lam*imgs + (1-lam)*imgs[perm_index, :]
        mix_masks = masks[perm_index]
        
        # we need lambda for backpropagating
        return mixed_imgs, masks, mix_masks, lam
    """

# data loader for dataset
def get_loader(data_set, batch_size=64, shuffle=True, num_workers=8):
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader

# data loader with mixup
def get_mixup_loader(data_set, alpha=0.2, mixup_transform=None, batch_size=64, shuffle=True, num_workers=8):
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=Mixup(alpha, mixup_transform))

    return data_loader

# data loader with mixup in segmentation
def get_mixup_segmentation_loader(data_set, alpha=0.2, mixup_transform=None, batch_size=64, shuffle=True, num_workers=8):
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=SegmentationMixup(alpha, mixup_transform))

    return data_loader
