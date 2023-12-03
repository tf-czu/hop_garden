"""
    Convert pixel data to new resolution.
"""

import json
import numpy as np


def confert_config(config, size, name):
    plot_cnt_or = np.array(config["plot_cnt"])
    section_or = np.array(config["rot_rec"])
    x_or, y_or = config["im-size"]

    x_new, y_new = size.split()
    x_scale = float(x_new)/x_or
    y_scale = float(y_new)/y_or
    scale = np.array([x_scale, y_scale])

    plot_cnt_new = np.uint32(np.round(plot_cnt_or*scale))
    section_new = np.uint32(np.round(section_or*scale))

    ret_dic = config
    ret_dic["plot_cnt"] = plot_cnt_new.tolist()
    ret_dic["rot_rec"] = section_new.tolist()
    ret_dic["im-size"] = [int(x_new), int(y_new)]
    ret_dic["px-size"] = config["px-size"]/x_scale

    with open(name, 'w') as outfile:
        json.dump(ret_dic, outfile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--im-size', help='Sizes for the new config.')
    parser.add_argument('--name', help='Config name.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    size = args.im_size
    name = args.name
    confert_config(config, size, name)