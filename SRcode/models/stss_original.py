import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


class LWGatedConv2D(nn.Module):
    def __init__(self, in_cha, out_cha, kernel, stride, pad):
        super(LWGatedConv2D, self).__init__()

        self.conv_feature = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=kernel, stride=stride, padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=in_cha, out_channels=1, kernel_size=kernel, stride=stride, padding=pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv_feature(x)
        mask = self.conv_mask(x)

        return x1 * mask


class DownLWGated(nn.Module):
    def __init__(self, in_cha, out_cha):
        super().__init__()
        self.down_sample = LWGatedConv2D(in_cha, in_cha, kernel=3, stride=2, pad=1)
        self.conv_1 = LWGatedConv2D(in_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = LWGatedConv2D(out_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_cha, out_cha, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                nn.Conv2d(in_cha, in_cha // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_cha // 2, out_cha, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_cha, in_cha // 2, kernel_size=2, stride=2, padding=1)
            self.conv = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_cha, out_cha, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class StssOriginal(BaseModel):
    def __init__(self, scale: int, buffer_cha: int = 0, history_frames: int = 2):
        super(StssOriginal, self).__init__(scale=scale, down_and_up=3)

        self.conv_in = nn.Sequential(
            LWGatedConv2D(3 + buffer_cha, 24, kernel=3, stride=1, pad=1),
            nn.ReLU(inplace=True),
            LWGatedConv2D(24, 24, kernel=3, stride=1, pad=1),
            nn.ReLU(inplace=True)
        )

        self.down_1 = DownLWGated(24, 24)
        self.down_2 = DownLWGated(24, 32)
        self.down_3 = DownLWGated(32, 32)

        self.his_1 = nn.Sequential(
            nn.Conv2d(history_frames * 3, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.his_2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.his_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # for now we don't use it because we don't have the same warped image out the custom engine:
        # https://github.com/fuxihao66/UnrealEngine/tree/5.1
        # self.latentEncoder = nn.Sequential(
        #     nn.Conv2d(32 + 10, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1, padding=1, bias=True)
        # )
        # self.KEncoder = nn.Sequential(
        #     nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True)
        # )
        
        self.up_1 = Up(64 + 32, 32)
        self.up_2 = Up(56, 24)
        self.up_3 = Up(48, 24)

        self.conv_out = nn.Sequential(
            nn.Conv2d(24, 3 * self.scale ** 2, kernel_size=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x, his): #feature, his):
        # x = torch.cat([x, feature], 1)
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bilinear")
        x1 = self.conv_in(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        his = torch.cat(torch.unbind(his, 1), 1)
        his = self.his_1(his)
        his = self.his_2(his)
        his = self.his_3(his)

        # inpaint_feature = self.hole_inpaint(x4, mask=?, feature)

        x4 = torch.cat([x4, his], 1)

        x = self.up_1(x4, x3)
        x = self.up_2(x, x2)
        x = self.up_3(x, x1)
        x = self.conv_out(x)
        x = x + x_up
        return x

    # def hole_inpaint(self, x, mask, feature):
    #     x_down = x
    #     mask_down = F.interpolate(mask, scale_factor=0.125, mode='bilinear')
    #     feature_down = F.interpolate(feature, scale_factor=0.125, mode='bilinear')
    #
    #     latent_code = self.latentEncoder(torch.cat([x_down, feature_down], dim=1)) * mask_down
    #     K_map = F.normalize(self.KEncoder(feature_down), p=2, dim=1)
    #
    #     b, c, h, w = list(K_map.size())
    #     md = 2
    #     f1 = F.unfold(K_map * mask_down, kernel_size=(2 * md + 1, 2 * md + 1), padding=(md, md), stride=(1, 1))
    #     f1 = f1.view([b, c, -1, h, w])
    #     f2 = K_map.view([b, c, 1, h, w])
    #     weight_k = torch.relu((f1 * f2).sum(dim=1, keepdim=True))
    #
    #     b, c, h, w = list(latent_code.size())
    #     v = F.unfold(latent_code, kernel_size=(2 * md + 1, 2 * md + 1), padding=(md, md), stride=(1, 1))
    #     v = v.view([b, c, -1, h, w])
    #
    #     agg_latent = (v * weight_k).sum(dim=2) / (weight_k.sum(dim=2).clamp_min(1e-6))
    #     return agg_latent


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StssOriginal(scale=2, ).to(device)
    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    # feature = (batch_size, 9, 1920, 1080)
    his = (batch_size, 3, 2, 1920, 1080)
    input_size = (input_data, his) # feature, his)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
