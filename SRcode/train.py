import argparse
from typing import Union

from lpips import lpips
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from data.dataloader import SingleImagePair, MultiImagePair
from config import load_yaml_into_config, create_comment_from_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SR network based on a config file.")
    parser.add_argument('file_path', type=str, nargs='?', default='configs/RTSRN/rtsrn_01.yaml', help="Path to the config file")
    args = parser.parse_args()
    return args


def save_model(filename: str, model: nn.Module) -> None:
    model_path = "pretrained_models/" + filename + ".pth"
    torch.save(model.state_dict(), model_path)


def write_frames(writer: SummaryWriter, step: int, extra: bool, buffers: dict[str, bool],
                 input_t0: list[torch.Tensor], output_t0: torch.Tensor, gt_t0: torch.Tensor,
                 input_t1: list[torch.Tensor], output_t1: torch.Tensor, gt_t1: torch.Tensor) -> None:
    # SS
    # LR
    writer.add_images("SS/LR", input_t0[0], step)
    # Buffers
    buffer_keys = []
    for key, val in buffers.items():
        if val:
            buffer_keys.append(key)
    for i, buffer in enumerate(input_t0[2]):
        writer.add_images(f"SS/{buffer_keys[i]}", buffer, step)
    # History
    for i, history_frame in enumerate(input_t0[3]):
        writer.add_images(f"SS/History T - {(i+1)*2}", history_frame, step)
    # Output
    writer.add_images("SS/Output", output_t0, step)
    # GT
    writer.add_images("SS/GT", gt_t0, step)

    if not extra: # if we do not do extrapolation we will not see this here
        return
    # ESS
    # LR
    writer.add_images("ESS/LR", input_t1[0], step)
    # Mask
    writer.add_images("ESS/Mask", 1 - input_t1[1], step)
    # Buffers -> same buffer keys
    for i, buffer in enumerate(input_t1[2]):
        writer.add_images(f"ESS/{buffer_keys[i]}", buffer, step)
    # History
    for i, history_frame in enumerate(input_t1[3]):
        writer.add_images(f"ESS/History T - {(i + 1) * 2}", history_frame, step)
    # Output
    writer.add_images("ESS/Output", output_t1, step)
    # GT
    writer.add_images("ESS/GT", gt_t1, step)


def write_vsr_frames(writer: SummaryWriter, step: int, lr_image: torch.Tensor, history: list[torch.Tensor],
                     output: torch.Tensor, gt: torch.Tensor) -> None:
    # LR
    writer.add_images("Images/LR", lr_image, step)
    # History
    if history:
        for i, history_frame in enumerate(history):
            writer.add_images(f"Images/History T-{(i + 1) * 2}", history_frame, step)
    # Output
    writer.add_images("Images/Output", output, step)
    # GT
    writer.add_images("Images/GT", gt, step)


def write_vsr_frames_with_buffers(writer: SummaryWriter, step: int, buffer_dict: dict[str, bool], lr_image: torch.Tensor,
                     buffers: list[torch.Tensor], history: list[torch.Tensor], masks: list[torch.Tensor],
                     output: torch.Tensor, gt: torch.Tensor) -> None:
    # LR
    writer.add_images("Images/LR", lr_image, step)
    # Buffers
    buffer_keys = []
    for key, val in buffer_dict.items():
        if val:
            buffer_keys.append(key)
    for i, buffer in enumerate(buffers):
        writer.add_images(f"Images/Buffer {buffer_keys[i]}", buffer, step)
    # History
    for i, history_frame in enumerate(history):
        writer.add_images(f"Images/History T-{(i + 1) * 2}", history_frame, step)
    # Masks
    for i, mask in enumerate(masks):
        writer.add_images(f"Images/Mask T-{(i + 1) * 2}", mask, step)
    # Output
    writer.add_images("Images/Output", output, step)
    # GT
    writer.add_images("Images/GT", gt, step)


def train(filepath: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = load_yaml_into_config(filepath)
    print(config)
    filename = config.filename.split('.')[0]
    writer = SummaryWriter(filename_suffix=create_comment_from_config(config), comment=filename)  # log_dir="runs"
    # Hyperparameters
    batch_size = config.batch_size
    epochs = config.epochs
    num_workers = config.number_workers
    start_decay_epoch = config.start_decay_epoch

    # Model details
    model = config.model.to(device)
    extra = config.extra
    buffers = config.buffers
    criterion = config.criterion
    optimizer = config.optimizer
    scheduler = config.scheduler

    # Loading and preparing data
    # train data
    train_dataset = config.train_dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # val data
    val_dataset = config.val_dataset
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    writer.add_text("Model detail", f"{config}", -1)

    iteration_counter = 0

    # Training & Validation Loop
    for epoch in tqdm(range(epochs), desc='Train & Validate', dynamic_ncols=True):
        # train loop
        ss_total_loss = 0.0
        ess_total_loss = 0.0
        total_loss = 0.0

        for ss, ess in tqdm(train_loader, desc=f'Training, Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            # setup
            optimizer.zero_grad()

            # prepare data
            # SS
            ss_lr_image = ss[0].to(device) # shared
            ss_mask = ss[1].to(device)
            ss_feature_images = [img.to(device) for img in ss[2]]
            if ss_feature_images:
                ss_feature_images = torch.cat(ss_feature_images, dim=1)
            history_images = [img.to(device) for img in ss[3]]
            if history_images:
                history_images = torch.stack(history_images, dim=2) # shared
            ss_hr_image = ss[4].to(device)
            # ESS
            if extra:
                ess_lr_image = ess[0].to(device)
                ess_mask = ess[1].to(device)
                ess_feature_images = [img.to(device) for img in ess[2]]
                if ess_feature_images:
                    ess_feature_images = torch.cat(ess_feature_images, dim=1)
                ess_hr_image = ess[4].to(device)

            # forward pass
            # depending on the network we perform two forward passes or only one
            if model.do_two:
                # SS
                ss_output = model(ss_lr_image, ss_mask, ss_feature_images, history_images)
                ss_loss = criterion(ss_output, ss_hr_image) # + 0.1 * lpips
                ss_loss.backward(retain_graph=extra) # accumulate gradients for SS
                # ESS
                ess_loss = 0
                if extra:
                    ess_output = model(ess_lr_image, ess_mask, ess_feature_images, history_images)
                    ess_loss = criterion(ess_output, ess_hr_image) # + 0.1* lpips
                    ess_loss.backward() # accumulate gradients for ESS
                loss = ss_loss + ess_loss
            else: # TODO
                if extra:
                    lr_images = torch.cat([ss_lr_image, ess_lr_image], 1)
                    if torch.is_tensor(ss_feature_images):
                        feature_images = torch.cat([ss_feature_images, ess_feature_images], 1)
                    else:
                        feature_images = []
                    hr_images = torch.cat([ss_hr_image, ess_hr_image], 1)
                else:
                    lr_images = ss_lr_image
                    feature_images = ss_feature_images
                    hr_images = ss_hr_image
                output = model(lr_images, feature_images, history_images)
                if extra:
                    output = torch.cat(output, dim=1)
                loss = criterion(output, hr_images)
                loss.backward()

            optimizer.step()

            if model.do_two:
                ss_total_loss += ss_loss.item()
                ess_total_loss += ess_loss.item()
            total_loss += loss.item()
            iteration_counter += 1 * batch_size

        # scheduler update if we have one
        if scheduler is not None:
            if epoch > start_decay_epoch:
                scheduler.step()
        # Loss
        average_ss_loss = ss_total_loss / len(train_loader)
        average_ess_loss = ess_total_loss / len(train_loader)
        average_loss = total_loss / len(train_loader)
        print("\n")
        if model.do_two:
            print(f"SS Loss: {average_ss_loss:.4f}, ESS Loss: {average_ess_loss:.4f}, Loss: {average_loss:.4f}\n")
            if extra:
                writer.add_scalar('Train/SS Loss', average_ss_loss, epoch)
                writer.add_scalar('Train/ESS Loss', average_ess_loss, epoch)
        else:
            print(f"Loss: {average_loss:.4f}\n")
        # Log loss to TensorBoard
        writer.add_scalar('Train/Loss', average_loss, epoch)

        # val loop
        if (epoch + 1) % 5 != 0:
            continue
        total_ss_metrics = utils.Metrics([0], [0])  # TODO: abstract number of values based on second dim of tensor [8, 2, 3, 1920, 1080]
        total_ess_metrics = utils.Metrics([0], [0])

        val_counter = 0
        for ss, ess in tqdm(val_loader, desc=f"Validation, Epoch {epoch + 1}/{epochs}", dynamic_ncols=True):
            # prepare data
            # SS
            ss_lr_image = ss[0].to(device)  # shared
            ss_mask = ss[1].to(device)
            ss_feature_images = [img.to(device) for img in ss[2]]
            if ss_feature_images:
                ss_feature_images = torch.cat(ss_feature_images, dim=1)
            history_images = [img.to(device) for img in ss[3]]
            if history_images:
                history_images = torch.stack(history_images, dim=2)  # shared
            ss_hr_image = ss[4].to(device)
            # ESS
            if extra:
                ess_lr_image = ess[0].to(device)
                ess_mask = ess[1].to(device)
                ess_feature_images = [img.to(device) for img in ess[2]]
                if ess_feature_images:
                    ess_feature_images = torch.cat(ess_feature_images, dim=1)
                ess_hr_image = ess[4].to(device)

            with torch.no_grad():
                # forward pass
                # depending on the network we perform two forward passes or only one
                if model.do_two:
                    # SS
                    ss_output = model(ss_lr_image, ss_mask, ss_feature_images, history_images)
                    ss_output = torch.clamp(ss_output, min=0.0, max=1.0)
                    # ESS
                    if extra:
                        ess_output = model(ess_lr_image, ess_mask, ess_feature_images, history_images)
                        ess_output = torch.clamp(ess_output, min=0.0, max=1.0)
                else: # TODO
                    if extra:
                        lr_images = torch.cat([ss_lr_image, ess_lr_image], 1)
                        if torch.is_tensor(ss_feature_images):
                            feature_images = torch.cat([ss_feature_images, ess_feature_images], 1)
                        else:
                            feature_images = []
                    else:
                        lr_images = ss_lr_image
                        feature_images = ss_feature_images
                    if extra:
                        ss_output, ess_output = model(lr_images, feature_images, history_images)
                        ss_output = torch.clamp(ss_output, min=0.0, max=1.0)
                        ess_output = torch.clamp(ess_output, min=0.0, max=1.0)
                    else:
                        ss_output = model(lr_images, feature_images, history_images)
                        ss_output = torch.clamp(ss_output, min=0.0, max=1.0)

            # Calc PSNR and SSIM
            # SS frame
            ss_metric = utils.calculate_metrics(ss_hr_image, ss_output, "single")
            total_ss_metrics += ss_metric
            if extra:
                # ESS frame
                ess_metric = utils.calculate_metrics(ess_hr_image, ess_output, "single")
                total_ess_metrics += ess_metric

            # Display the val process in tensorboard
            if val_counter != 0:
                continue
            write_frames(writer, iteration_counter, extra, buffers, ss, ss_output, ss_hr_image, ess, ess_output, ess_hr_image)
            val_counter += 1

        # PSNR & SSIM
        average_ss_metric = total_ss_metrics / len(val_loader)
        if extra:
            average_ess_metric = total_ess_metrics / len(val_loader)
            average_metric = (average_ss_metric + average_ess_metric) / 2
        else:
            average_metric = average_ss_metric
        print("\n")
        if extra:
            print(f"SS {average_ss_metric}")
            print(f"ESS {average_ess_metric}")
        print(f"Total {average_metric}")

        # Log PSNR & SSIM to TensorBoard
        # PSNR
        if extra:
            writer.add_scalar("Val/PSNR/SS", average_ss_metric.average_psnr, epoch)
            writer.add_scalar("Val/PSNR/ESS", average_ess_metric.average_psnr, epoch)
        writer.add_scalar("Val/PSNR/Average", average_metric.average_psnr, epoch)
        # SSIM
        if extra:
            writer.add_scalar("Val/SSIM/SS", average_ss_metric.average_ssim, epoch)
            writer.add_scalar("Val/SSIM/ESS", average_ess_metric.average_ssim, epoch)
        writer.add_scalar("Val/SSIM/Average", average_metric.average_ssim, epoch)

    # End Log
    writer.close()
    # Save trained models
    save_model(filename, model)


def train2(filepath: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = load_yaml_into_config(filepath)
    print(config)
    filename = config.filename.split('.')[0]
    writer = SummaryWriter(filename_suffix=create_comment_from_config(config), comment=filename)  # log_dir="runs"
    # Hyperparameters
    batch_size = config.batch_size
    epochs = config.epochs
    num_workers = config.number_workers
    start_decay_epoch = config.start_decay_epoch

    # Model details
    model = config.model.to(device)
    buffers_dict = config.buffers
    criterion = config.criterion
    optimizer = config.optimizer
    scheduler = config.scheduler

    # Loading and preparing data
    # train data
    train_dataset = config.train_dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # val data
    val_dataset = config.val_dataset
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    writer.add_text("Model detail", f"{config}", -1)

    iteration_counter = 0

    # Training & Validation Loop
    for epoch in tqdm(range(epochs), desc='Train & Validate', dynamic_ncols=True):
        # train loop
        total_loss = 0.0

        for ss in tqdm(train_loader, desc=f'Training, Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            # setup
            optimizer.zero_grad()
            # prepare data
            lr_image = ss[0].to(device)  # shared
            feature_images = [img.to(device) for img in ss[1]]
            if feature_images:
                feature_images = torch.cat(feature_images, dim=1)
            history_images = [img.to(device) for img in ss[2]]
            if history_images:
                history_images = torch.stack(history_images, dim=2)
            mask_images = [img.to(device) for img in ss[3]]
            if mask_images:
                mask_images = torch.stack(mask_images, dim=2)
            hr_image = ss[4].to(device)

            # forward pass
            output = model(lr_image) #feature_images, history_images, mask_images)
            loss = criterion(output, hr_image)  # + 0.1 * lpips
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            iteration_counter += 1 * batch_size

        # scheduler update if we have one
        if scheduler is not None:
            if epoch > start_decay_epoch:
                scheduler.step()
        # Loss
        average_loss = total_loss / len(train_loader)
        print("\n")
        print(f"Loss: {average_loss:.4f}\n")
        # Log loss to TensorBoard
        writer.add_scalar('Train/Loss', average_loss, epoch)

        # val loop
        if (epoch + 1) % 5 != 0:
            continue
        total_metrics = utils.Metrics([0], [0])

        val_counter = 0
        for ss in tqdm(val_loader, desc=f"Validation, Epoch {epoch + 1}/{epochs}", dynamic_ncols=True):
            # prepare data
            lr_image = ss[0].to(device)  # shared
            feature_images = [img.to(device) for img in ss[1]]
            if feature_images:
                feature_images = torch.cat(feature_images, dim=1)
            history_images = [img.to(device) for img in ss[2]]
            if history_images:
                history_images = torch.stack(history_images, dim=2)
            mask_images = [img.to(device) for img in ss[3]]
            if mask_images:
                mask_images = torch.stack(mask_images, dim=2)
            hr_image = ss[4].to(device)

            with torch.no_grad():
                # forward pass
                output = model(lr_image)#, feature_images, history_images, mask_images)
                output = torch.clamp(output, min=0.0, max=1.0)

            # Calc PSNR and SSIM
            # SS frame
            metric = utils.calculate_metrics(hr_image, output, "single")
            total_metrics += metric

            # Display the val process in tensorboard
            if val_counter != 0:
                continue
            write_vsr_frames(writer, iteration_counter, buffers_dict, ss[0], ss[1], ss[2], ss[3],
                             output, ss[4])
            val_counter += 1

        # PSNR & SSIM
        average_metric = total_metrics / len(val_loader)
        print("\n")
        print(f"Total {average_metric}")

        # Log PSNR & SSIM to TensorBoard
        # PSNR
        writer.add_scalar("Val/PSNR", average_metric.average_psnr, epoch)
        # SSIM
        writer.add_scalar("Val/SSIM", average_metric.average_ssim, epoch)

    # End Log
    writer.close()
    # Save trained models
    save_model(filename, model)


def train3(filepath: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = load_yaml_into_config(filepath)
    print(config)
    filename = config.filename.split('.')[0]
    writer = SummaryWriter(filename_suffix=create_comment_from_config(config), comment=filename)  # log_dir="runs"
    # Hyperparameters
    batch_size = config.batch_size
    epochs = config.epochs
    num_workers = config.number_workers
    start_decay_epoch = config.start_decay_epoch

    # Model details
    model = config.model.to(device)
    scaler = amp.GradScaler()
    # buffers_dict = config.buffers
    criterion = config.criterion
    optimizer = config.optimizer
    scheduler = config.scheduler

    # Loading and preparing data
    # train data
    train_dataset = config.train_dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # val data
    val_dataset = config.val_dataset
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    eval_alex_model = lpips.LPIPS(net='alex').cuda()

    writer.add_text("Model detail", f"{config}", -1)

    iteration_counter = 0

    # Training & Validation Loop
    for epoch in tqdm(range(epochs), desc='Train & Validate', dynamic_ncols=True):
        # train loop
        total_loss = 0.0

        model.reset() # TODO not a good idea, prev state needs to be handeld from dataloader or train script, else we get a problem with the batch dim!

        for lr_image, history_images, hr_image in tqdm(train_loader, desc=f'Training, Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            # setup
            optimizer.zero_grad()
            # prepare data
            lr_image = lr_image.to(device)
            if history_images: # in case we use 0 prev frames
                history_images = [img.to(device) for img in history_images]
                history_images = torch.stack(history_images, dim=1)
            hr_image = hr_image.to(device)

            # forward pass
            with amp.autocast():
                if history_images:
                    output = model(lr_image, history_images)
                else:
                    output = model(lr_image)
                loss = criterion(output, hr_image)
            scaler.scale(loss.mean()).backward()

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            iteration_counter += 1 * batch_size

        # scheduler update if we have one
        if scheduler is not None:
            if epoch > start_decay_epoch:
                scheduler.step()
        # Loss
        average_loss = total_loss / len(train_loader)
        print("\n")
        print(f"Loss: {average_loss:.4f}\n")
        # Log loss to TensorBoard
        writer.add_scalar('Train/Loss', average_loss, epoch)

        # val loop
        if (epoch + 1) % 10 != 0:
            continue
        total_metrics = utils.Metrics(0, 0, 0)

        val_counter = 0
        model.reset()
        for lr_image, history_images, hr_image in tqdm(val_loader, desc=f"Validation, Epoch {epoch + 1}/{epochs}", dynamic_ncols=True):
            # prepare data
            lr_image = lr_image.to(device)
            if history_images:
                history_images = [img.to(device) for img in history_images]
                history_images = torch.stack(history_images, dim=1)
            hr_image = hr_image.to(device)

            with torch.no_grad():
                # forward pass
                with amp.autocast():
                    if history_images:
                        output = model(lr_image, history_images)
                    else:
                        output = model(lr_image)
                output = torch.clamp(output, min=0.0, max=1.0)

            # Calc PSNR and SSIM
            # SS frame
            metric = utils.calculate_metrics(hr_image.squeeze(0), output.squeeze(0), eval_alex_model)
            total_metrics += metric

            # Display the val process in tensorboard
            if val_counter != 0:
                continue
            if history_images:
                history_images = torch.unbind(history_images, dim=1)
            write_vsr_frames(writer, iteration_counter, lr_image, history_images, output, hr_image)
            val_counter += 1

        # PSNR & SSIM
        average_metric = total_metrics / len(val_loader)
        print("\n")
        print(f"Total {average_metric}")

        # Log PSNR & SSIM to TensorBoard
        # PSNR
        writer.add_scalar("Val/PSNR", average_metric.psnr_value, epoch)
        # SSIM
        writer.add_scalar("Val/SSIM", average_metric.ssim_value, epoch)
        # LPIPS
        writer.add_scalar("Val/LPIPS", average_metric.lpips_value, epoch)

        # End Log
    writer.close()
    # Save trained models
    save_model(filename, model)


def main() -> None:
    args = parse_arguments()
    file_path = args.file_path
    train3(file_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
