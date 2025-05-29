from __future__ import division
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import math

from machine_perception.experiments.stm.helpers import pad_divide_by, to_cuda


class ResBlock(nn.Module):
    def __init__(self, indim: int, outdim: int | None = None, stride: int = 1):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(
                indim, outdim, kernel_size=3, padding=1, stride=stride
            )

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        # self.register_buffer(
        #     "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # )
        # self.register_buffer(
        #     "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # )
        self.mean = nn.Parameter(
            torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            requires_grad=False,
        )
        self.std = nn.Parameter(
            torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            requires_grad=False,
        )

    def forward(self, in_f: torch.Tensor, in_m: torch.Tensor, in_o: torch.Tensor):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float()  # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Encoder_Q(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer(
            "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, in_f: torch.Tensor):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes: int, planes: int, scale_factor: float = 2.0):
        super().__init__()
        self.convFS = nn.Conv2d(
            inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1
        )
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(
            pm, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super().__init__()
        self.convFM = nn.Conv2d(
            1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1
        )
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, scale_factor=4, mode="bilinear", align_corners=False)
        return p  # , p2, p3, p4


class Memory(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        m_in: torch.Tensor,
        m_out: torch.Tensor,
        q_in: torch.Tensor,
        q_out: torch.Tensor,
    ):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H * W)  # b, emb, HW

        p = torch.bmm(mi, qi)  # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW

        mo = m_out.view(B, D_o, T * H * W)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim: int, keydim: int, valdim: int):
        super().__init__()
        self.Key = nn.Conv2d(
            indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1
        )
        self.Value = nn.Conv2d(
            indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.Key(x), self.Value(x)


class STM(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)

    def pad_memory(
        self, mems: list[torch.Tensor], num_objects: int, K: int
    ) -> list[torch.Tensor]:
        """Shape of each `mem` in `mems`: `(num_objects, 128 and 512, H/16, W/16)`"""
        pad_mems = []
        for mem in mems:
            pad_mem = to_cuda(
                torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3])
            )  # (1, K, 128 and 512, 1, H/16, W/16)
            pad_mem[0, 1 : num_objects + 1, :, 0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(
        self, frame: torch.Tensor, masks: torch.Tensor, num_objects: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # memorize a frame
        num_objects: int = num_objects[0].item()
        _, K, H, W = masks.shape  # B = 1

        (frame, masks), pad = pad_divide_by(
            [frame, masks], 16, (frame.size()[2], frame.size()[3])
        )

        # make batch arg list
        B_list = {"f": [], "m": [], "o": []}
        for o in range(1, num_objects + 1):  # 1 - number_objects
            B_list["f"].append(frame)
            B_list["m"].append(masks[:, o])
            B_list["o"].append(
                (
                    torch.sum(masks[:, 1:o], dim=1)
                    + torch.sum(masks[:, o + 1 : num_objects + 1], dim=1)
                ).clamp(0, 1)
            )

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_["f"], B_["m"], B_["o"])
        k4, v4 = self.KV_M_r4(r4)  # (num_objects, 128 and 512, H/16, W/16)
        k4, v4 = self.pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def soft_aggregation(self, ps: torch.Tensor, K: int):
        num_objects, H, W = ps.shape
        em = to_cuda(torch.zeros(1, K, H, W))
        em[0, 0] = torch.prod(1 - ps, dim=0)  # bg prob
        em[0, 1 : num_objects + 1] = ps  # obj prob
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def segment(
        self,
        frame: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        num_objects: torch.Tensor,
    ):
        num_objects: int = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape  # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16

        # expand to ---  num_objects, c, h, w
        k4e, v4e = (
            k4.expand(num_objects, -1, -1, -1),
            v4.expand(num_objects, -1, -1, -1),
        )
        r3e, r2e = (
            r3.expand(num_objects, -1, -1, -1),
            r2.expand(num_objects, -1, -1, -1),
        )

        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(
            keys[0, 1 : num_objects + 1], values[0, 1 : num_objects + 1], k4e, v4e
        )
        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:, 1]  # num_objects, h, w
        # ps = indipendant possibility to belong to each object

        logit = self.soft_aggregation(ps, K)  # 1, K, H, W

        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2] : -pad[3], :]
        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0] : -pad[1]]

        return logit

    def forward(self, frame, keys, values, masks, num_objects):
        if keys.dim() > 4:  # keys
            return self.segment(frame, keys, values, num_objects)
        else:
            return self.memorize(frame, masks, num_objects)

    # def forward(self, *args, **kwargs):
    #     if args[1].dim() > 4:  # keys
    #         return self.segment(*args, **kwargs)
    #     else:
    #         return self.memorize(*args, **kwargs)


def load_stm_state_dict(weights_path: str = "STM_weights.pth") -> dict[str, Any]:
    state_dict = torch.load(weights_path, weights_only=True)
    new_state_dict = {key.lstrip("module."): val for key, val in state_dict.items()}
    return new_state_dict


if __name__ == "__main__":
    stm = STM()

    # new_state_dict = load_stm_state_dict("STM_weights.pth")
    # stm.load_state_dict(new_state_dict)
    # print(stm)

    K = 5
    H, W = 720, 480
    d = 16
    H_emb, W_emb = 720 // d, 480 // d
    T = 3
    K = 5
    KEY_DIM, VAL_DIM = 128, 512

    example_inputs = [
        torch.rand([1, 3, H, W]),  # frame
        torch.rand([1, K, KEY_DIM, T, H_emb, W_emb]),  # keys: [1, 5, 128, 3, 45, 30]
        torch.rand([1, K, VAL_DIM, T, H_emb, W_emb]),  # values: [1, 5, 512, 3, 45, 30]
        torch.rand([1, K, H, W]),  # masks
        torch.tensor([3], dtype=int),  # num_objects
    ]

    result = stm(*example_inputs)
    print(result)

    # model_scripted = torch.jit.script(stm)
    # torch.save(model_scripted, "stm_scripted.pt")
    # torch.save(stm, "stm.pt")
