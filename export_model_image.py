import torch
from torchview import draw_graph
from model.LPRNet import build_lprnet

model = build_lprnet(class_num=37)
# state_dict = torch.load("Final_LPRNet_model.pth", map_location="cpu")
# model.load_state_dict(state_dict)

graph = draw_graph(model, input_size=(1, 3, 24, 94), expand_nested=True)
graph.visual_graph.render("lprnet_architecture", format="png")  # saves a diagram
