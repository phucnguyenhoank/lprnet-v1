import torch
from model.LPRNet import build_lprnet

model = build_lprnet(class_num=37)
dummy = torch.randn(1, 3, 24, 94)

torch.onnx.export(
    model,
    dummy,
    "lprnet.onnx",
    opset_version=20,   # or 17 if you want max compatibility
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
