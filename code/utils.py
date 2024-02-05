import torchvision.transforms.functional as F


def pad_to_divisible(image_tensor, factor):
    _, _, height, width = image_tensor.size()
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    padded_image = F.pad(image_tensor, [0, 0, pad_width, pad_height], padding_mode="edge")
    return padded_image


def pad_or_crop_to_target(image_tensor, target_tensor):
    _, height, width = image_tensor.size()
    _, target_height, target_width = target_tensor.size()
    if height < target_height:
        pad_height = max(0, target_height - height)
        image_tensor = F.pad(image_tensor, [0, 0, 0, pad_height], padding_mode="edge")
    else:
        image_tensor = image_tensor[:, :target_height, :]
    if width < target_width:
        pad_width = max(0, target_width - width)
        image_tensor = F.pad(image_tensor, [0, 0, pad_width, 0], padding_mode="edge")
    else:
        image_tensor = image_tensor[:, :, :target_width]
    return image_tensor


