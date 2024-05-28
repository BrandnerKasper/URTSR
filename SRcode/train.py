import argparse
from typing import Union

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from data.dataloader import SingleImagePair, MultiImagePair
from config import load_yaml_into_config, create_comment_from_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SR network based on a config file.")
    parser.add_argument('file_path', type=str, nargs='?', default='configs/stss2.yaml', help="Path to the config file")
    args = parser.parse_args()
    return args


def save_model(filename: str, model: nn.Module) -> None:
    model_path = "pretrained_models/" + filename + ".pth"
    torch.save(model.state_dict(), model_path)


def write_images(writer: SummaryWriter, mode: str, input_t: Union[torch.Tensor, list[torch.Tensor]], output_t: Union[torch.Tensor, list[torch.Tensor]], gt_t: Union[torch.Tensor, list[torch.Tensor]], step: int) -> None:
    match mode:
        case "single": # -> for Spatial SR
            # Display input tensors
            writer.add_image(f"Images/Input", input_t, step)
            # Display output tensors
            writer.add_image(f"Images/Output", output_t, step)
            # Display ground truth images
            writer.add_image(f"Images/GT", gt_t, step)
        case "multi": # -> for Temporal SR
            # Display input tensors
            for i in range(len(input_t)):
                writer.add_images(f"Images/Input_t{-i}", input_t[i], step)
            # Display output tensors
            for i in range(len(output_t)):
                writer.add_images(f"Images/Output_t{i}", output_t[i], step)
            # Display ground truth images
            for i in range(len(gt_t)):
                writer.add_images(f"Images/GT_t{i}", gt_t[i], step)


def write_stss_images(writer: SummaryWriter, step: int,
                      input_t0: list[torch.Tensor], output_t0: torch.Tensor, gt_t0: torch.Tensor,
                      input_t1: list[torch.Tensor], output_t1: torch.Tensor, gt_t1: torch.Tensor) -> None:
    # SS
    # LR
    writer.add_images("SS/LR", input_t0[0], step)
    # Features
    writer.add_images("SS/Basecolor", input_t0[1][0], step)
    writer.add_images("SS/depth", input_t0[1][1], step)
    writer.add_images("SS/metallic", input_t0[1][2], step)
    writer.add_images("SS/nov", input_t0[1][3], step)
    writer.add_images("SS/roughness", input_t0[1][4], step)
    writer.add_images("SS/velocity", input_t0[1][5], step)
    # History
    for i in range(len(input_t0[2])):
        writer.add_images(f"SS/History_t{-1-2*i}", input_t0[2][i], step)
    # Output
    writer.add_images("SS/Output", output_t0, step)
    # GT
    writer.add_images("SS/GT", gt_t0, step)

    # ESS
    # LR
    writer.add_images("ESS/LR", input_t1[0], step)
    # Features
    writer.add_images("ESS/Basecolor", input_t1[1][0], step)
    writer.add_images("ESS/depth", input_t1[1][1], step)
    writer.add_images("ESS/metallic", input_t1[1][2], step)
    writer.add_images("ESS/nov", input_t1[1][3], step)
    writer.add_images("ESS/roughness", input_t1[1][4], step)
    writer.add_images("ESS/velocity", input_t1[1][5], step)
    # History
    for i in range(len(input_t1[2])):
        writer.add_images(f"ESS/History_t{-2 - 2 * i}", input_t1[2][i], step)
    # Output
    writer.add_images("ESS/Output", output_t1, step)
    # GT
    writer.add_images("ESS/GT", gt_t1, step)


def write_cross_stss_images(writer: SummaryWriter, step: int,
                      input_t0: list[torch.Tensor], output_t0: torch.Tensor, gt_t0: torch.Tensor,
                      input_t1: list[torch.Tensor], output_t1: torch.Tensor, gt_t1: torch.Tensor) -> None:
    # SS
    # LR
    writer.add_images("SS/LR", input_t0[0], step)
    # Features
    writer.add_images("SS/Basecolor", input_t0[1][0], step)
    writer.add_images("SS/depth", input_t0[1][1], step)
    writer.add_images("SS/metallic", input_t0[1][2], step)
    writer.add_images("SS/roughness", input_t0[1][3], step)
    writer.add_images("SS/world_normal", input_t0[1][4], step)
    # History
    for i in range(len(input_t0[2])):
        writer.add_images(f"SS/History_t{-1-2*i}", input_t0[2][i], step)
    # Output
    writer.add_images("SS/Output", output_t0, step)
    # GT
    writer.add_images("SS/GT", gt_t0, step)

    # ESS
    # LR
    writer.add_images("ESS/LR", input_t1[0], step)
    # Features
    writer.add_images("ESS/Basecolor", input_t1[1][0], step)
    writer.add_images("ESS/depth", input_t1[1][1], step)
    writer.add_images("ESS/metallic", input_t1[1][2], step)
    writer.add_images("ESS/roughness", input_t1[1][3], step)
    writer.add_images("ESS/velocity", input_t1[1][4], step)
    # History
    for i in range(len(input_t1[2])):
        writer.add_images(f"ESS/History_t{-2 - 2 * i}", input_t1[2][i], step)
    # Output
    writer.add_images("ESS/Output", output_t1, step)
    # GT
    writer.add_images("ESS/GT", gt_t1, step)


def train(filepath: str) -> None:
    config = load_yaml_into_config(filepath)
    # based on which dataset we train, we decide if its SISR or STSS
    dataset_type = config.train_dataset
    if isinstance(dataset_type, SingleImagePair):
        train_single(filepath)
    elif isinstance(dataset_type, MultiImagePair):
        train_multi(filepath)
    else:
        train_stss(filepath)


def train_single(filepath: str) -> None:
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
        total_loss = 0.0

        for lr_image, hr_image in tqdm(train_loader, desc=f'Training, Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            optimizer.zero_grad()
            output = model(lr_image)
            loss = criterion(output, hr_image)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iteration_counter += 1

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
        for lr_image, hr_image in tqdm(val_loader, desc=f"Validation, Epoch {epoch + 1}/{epochs}", dynamic_ncols=True):
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            with torch.no_grad():
                output_image = model(lr_image)
                output_image = torch.clamp(output_image, min=0.0, max=1.0)
            # Calc PSNR and SSIM
            metrics = utils.calculate_metrics(hr_image, output_image, "single")
            total_metrics += metrics
            # Display the val process in tensorboard
            if val_counter != 0:
                continue
            write_images(writer, "single", lr_image, output_image, hr_image, iteration_counter)
            val_counter += 1

        # PSNR & SSIM
        average_metric = total_metrics / len(val_loader)
        print("\n")
        print(average_metric)
        # Log PSNR & SSIM to TensorBoard
        # PSNR
        writer.add_scalar(f"Val/PSNR", average_metric.average_psnr, epoch)
        # SSIM
        writer.add_scalar(f"Val/SSIM", average_metric.average_ssim, epoch)

    # End Log
    writer.close()
    # Save trained models
    save_model(filename, model)


def train_multi(filepath: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = load_yaml_into_config(filepath)
    print(config)
    filename = config.filename.split('.')[0]
    writer = SummaryWriter(filename_suffix=create_comment_from_config(config), comment=filename) #log_dir="runs"
    # Hyperparameters
    batch_size = config.batch_size
    epochs = config.epochs
    num_workers = config.number_workers
    start_decay_epoch = config.start_decay_epoch

    # Model details
    model = config.model.to(device)
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
        total_loss = 0.0

        for lr_images, hr_images in tqdm(train_loader, desc=f'Training, Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            lr_images = [img.to(device) for img in lr_images]
            lr_image = lr_images[0]
            hr_images = [img.to(device) for img in hr_images]
            history_images = torch.stack(lr_images, dim=2)
            optimizer.zero_grad()
            output = model(lr_image, history_images)
            hr_images = torch.cat(hr_images)
            output = torch.cat(output)
            loss = criterion(output, hr_images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iteration_counter += 1

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
        if (epoch + 1) % 5 != 1:
            continue
        total_metrics = utils.Metrics([0, 0], [0, 0]) # TODO: abstract number of values based on second dim of tensor [8, 2, 3, 1920, 1080]

        val_counter = 0
        for lr_images, hr_images in tqdm(val_loader, desc=f"Validation, Epoch {epoch + 1}/{epochs}", dynamic_ncols=True):
            lr_images = [img.to(device) for img in lr_images]
            lr_image = lr_images[0]
            hr_images = [img.to(device) for img in hr_images]
            history_images = torch.stack(lr_images, dim=2)
            with torch.no_grad():
                output_image = model(lr_image, history_images)
                output_image = [torch.clamp(img, min=0.0, max=1.0) for img in output_image]
            # Calc PSNR and SSIM
            metrics = utils.calculate_metrics(hr_images, output_image, "multi")
            total_metrics += metrics
            # Display the val process in tensorboard
            if val_counter != 0:
                continue
            write_images(writer, "multi", lr_images, output_image, hr_images, iteration_counter)
            val_counter += 1

        # PSNR & SSIM
        average_metric = total_metrics / len(val_loader)
        print("\n")
        print(average_metric)
        # Log PSNR & SSIM to TensorBoard
        # PSNR
        for i in range(len(average_metric.psnr_values)):
            writer.add_scalar(f"Val/PSNR/image_{i}", average_metric.psnr_values[i], epoch)
        writer.add_scalar("Val/PSRN/average", average_metric.average_psnr, epoch)
        # SSIM
        for i in range(len(average_metric.ssim_values)):
            writer.add_scalar(f"Val/SSIM/image_{i}", average_metric.ssim_values[i], epoch)
        writer.add_scalar("Val/SSIM/average", average_metric.average_ssim, epoch)

    # End Log
    writer.close()
    # Save trained models
    save_model(filename, model)


def train_stss(filepath: str) -> None:
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

        # We do a double strategy here: Two forward passes, one for SS, one for ESS frame
        for ss, ess in tqdm(train_loader, desc=f'Training, Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            # setup
            optimizer.zero_grad()

            # forward pass for SS
            lr_image = ss[0].to(device) # shared
            ss_feature_images = [img.to(device) for img in ss[1]]
            ss_feature_images = torch.cat(ss_feature_images, dim=1)
            history_images = [img.to(device) for img in ss[2]]
            history_images = torch.stack(history_images, dim=2) # shared
            ss_hr_image = ss[3].to(device)
            ss_output = model(lr_image, ss_feature_images, history_images)
            ss_loss = criterion(ss_output, ss_hr_image) # + 0.1 * lpips

            # forward pass for ESS
            ess_feature_images = [img.to(device) for img in ess[1]]
            ess_feature_images = torch.cat(ess_feature_images, dim=1)
            ess_hr_image = ess[3].to(device)
            ess_output = model(lr_image, ess_feature_images, history_images)
            ess_loss = criterion(ess_output, ess_hr_image) # + 0.1* lpips

            # New loss
            loss = ss_loss + ess_loss
            loss.backward()
            optimizer.step()

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
        print(f"SS Loss: {average_ss_loss:.4f}, ESS Loss: {average_ess_loss:.4f}, Loss: {average_loss:.4f}\n")
        # Log loss to TensorBoard
        writer.add_scalar('Train/SS Loss', average_ss_loss, epoch)
        writer.add_scalar('Train/ESS Loss', average_ess_loss, epoch)
        writer.add_scalar('Train/Combined Loss', average_loss, epoch)

        # val loop
        if (epoch + 1) % 5 != 0:
            continue
        total_ss_metrics = utils.Metrics([0], [0])  # TODO: abstract number of values based on second dim of tensor [8, 2, 3, 1920, 1080]
        total_ess_metrics = utils.Metrics([0], [0])

        val_counter = 0
        for ss, ess in tqdm(val_loader, desc=f"Validation, Epoch {epoch + 1}/{epochs}", dynamic_ncols=True):
            # forward pass for SS
            lr_image = ss[0].to(device) # shared
            ss_feature_images = [img.to(device) for img in ss[1]]
            ss_feature_images = torch.cat(ss_feature_images, dim=1)
            history_images = [img.to(device) for img in ss[2]]
            history_images = torch.stack(history_images, dim=2)  # shared
            ss_hr_image = ss[3].to(device)

            # forward pass for ESS
            ess_feature_images = [img.to(device) for img in ess[1]]
            ess_feature_images = torch.cat(ess_feature_images, dim=1)
            ess_hr_image = ess[3].to(device)

            with torch.no_grad():
                # SS frame
                ss_output = model(lr_image, ss_feature_images, history_images)
                ss_output = torch.clamp(ss_output, min=0.0, max=1.0)
                # ESS frame
                ess_output = model(lr_image, ess_feature_images, history_images)
                ess_output = torch.clamp(ess_output, min=0.0, max=1.0)

            # Calc PSNR and SSIM
            # SS frame
            ss_metric = utils.calculate_metrics(ss_hr_image, ss_output, "single")
            total_ss_metrics += ss_metric
            # ESS frame
            ess_metric = utils.calculate_metrics(ess_hr_image, ess_output, "single")
            total_ess_metrics += ess_metric

            # Display the val process in tensorboard
            if val_counter != 0:
                continue
            write_cross_stss_images(writer, iteration_counter, ss, ss_output, ss_hr_image, ess, ess_output, ess_hr_image)
            val_counter += 1

        # PSNR & SSIM
        average_ss_metric = total_ss_metrics / len(val_loader)
        average_ess_metric = total_ess_metrics / len(val_loader)
        average_metric = (average_ss_metric + average_ess_metric) / 2
        print("\n")
        print(f"SS {average_ss_metric}")
        print(f"ESS {average_ess_metric}")
        print(f"Total {average_metric}")
        # Log PSNR & SSIM to TensorBoard
        # PSNR
        writer.add_scalar("Val/PSNR/SS", average_ss_metric.average_psnr, epoch)
        writer.add_scalar("Val/PSNR/ESS", average_ess_metric.average_psnr, epoch)
        writer.add_scalar("Val/PSNR/Average", average_metric.average_psnr, epoch)
        # SSIM
        writer.add_scalar("Val/SSIM/SS", average_ss_metric.average_ssim, epoch)
        writer.add_scalar("Val/SSIM/ESS", average_ess_metric.average_ssim, epoch)
        writer.add_scalar("Val/SSIM/Average", average_metric.average_ssim, epoch)

    # End Log
    writer.close()
    # Save trained models
    save_model(filename, model)


def main() -> None:
    args = parse_arguments()
    file_path = args.file_path
    train(file_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
