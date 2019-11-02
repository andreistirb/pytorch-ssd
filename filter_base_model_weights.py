import torch
from collections import OrderedDict

def filter_basenet_weights(source, target=None):
    target = source.split(".")[0] + "_base_model.pth"
    state_dict = torch.load(source)

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith("base_net"):
            print(key)
            new_key = key.replace("base_net.", '')
            print(new_key)
            new_state_dict[new_key] = value

    torch.save(new_state_dict, target)

if __name__ == "__main__":
    filter_basenet_weights("models/mobilenet-v1-ssd-mp-0_675.pth")