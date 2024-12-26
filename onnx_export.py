from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
import yaml
from tap import Tap

from hifigan.generator import HifiganGenerator


class HifiganONNX(torch.nn.Module):
    def __init__(self, generator: HifiganGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.unsqueeze(0)
        wav = self.generator(mel)
        return wav.squeeze(0)


class ExportOnnxParser(Tap):
    model_path: Path
    out_path: Path
    model_type: Literal['full', 'generator'] = 'full'
    generator_config : Path = None

    def configure(self) -> None:
        self.add_argument("model_path")
        self.add_argument("out_path")


if __name__ == '__main__':
    args = ExportOnnxParser().parse_args()

    if args.model_type == 'full':

        state_dict = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = state_dict["generator"]["model"]
        # removing shitty prefix from state dict keys
        prefix_l = len("module.")
        state_dict = OrderedDict((k[prefix_l:],v) for k, v in state_dict.items())
        if args.generator_config is not None:
            with args.generator_config.open("r") as f:
                generator_config = yaml.safe_load(f)
            model = HifiganGenerator(**generator_config)
        else:
            model = HifiganGenerator()
        model.load_state_dict(state_dict=state_dict)
    else:
        torch.serialization.add_safe_globals([HifiganGenerator])
        model = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=False)
    model_onnx = HifiganONNX(model)

    mel = torch.empty(128, 300, dtype=torch.float32).uniform_(-10, 0).to(torch.device('cpu'))
    print("Input shape: ", mel.shape)
    sample_input = [mel]
    print("Converting to ONNX ...")

    torch.onnx.export(model_onnx,
                      f=args.out_path,
                      args=tuple(sample_input),
                      # opset_version=args.onnx_opset,
                      input_names=["x"],
                      output_names=["wav"],
                      dynamic_axes={
                          "x": {1: "mel"},
                          "wav": {0: "samples"}
                      })
