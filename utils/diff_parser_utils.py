"""
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
from argparse import ArgumentParser


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    data_options(parser)
    diffusion_options(parser)
    model_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args_to_overwrite += ["data_root"]

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    print(args_path)
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            if a == "timestep_respacing" or a == "partial":
                continue
            setattr(args, a, model_args[a])

        elif "cond_mode" in model_args:  # backward compitability
            unconstrained = model_args["cond_mode"] == "no_cond"
            setattr(args, "unconstrained", unconstrained)

        else:
            print(
                "Warning: was not able to load [{}], using default value [{}] instead.".format(
                    a, args.__dict__[a]
                )
            )

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError("model_path argument must be specified.")



"""
--cuda：一个布尔类型的选项，用于指定是否使用CUDA设备（即GPU），默认为 True。
--device：一个整型选项，用于指定所使用的设备ID，默认为 0。
--seed：一个整型选项，用于设置随机种子，默认为 10。
--batch_size：一个整型选项，用于设置训练时的批量大小，默认为 64。
"""
def base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument(
        "--cuda", default=True, type=bool, help="Use cuda as default, if none then use CPU."
    )
    group.add_argument("--device", default=0, type=int, help="Device id")
    group.add_argument("--seed", default=10, type=int, help="fix random seed")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")

"""
--dataset 参数：
default="social"：如果命令行中没有提供该参数，则默认值为“social”。
choices=["social"]：这个参数的可选值列表。目前只有一个选项“social”。
--data_root 参数：
type=str：该参数的类型为字符串。
default=None：默认值为None。
--max_seq_length 参数：
default=600：如果命令行中没有提供该参数，则默认值为600。
--split 参数：
type=str：该参数的类型为字符串。
default=None：默认值为None。
choices=["test", "train", "val"]：这个参数的可选值列表，包括“test”、“train”和“val”。
"""
def data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default="social",
        choices=["social"],
        type=str,
        help="Dataset name (choose from list).",
    )
    group.add_argument("--data_root", type=str, default=None, help="dataset directory")
    group.add_argument("--max_seq_length", default=600, type=int)
    group.add_argument(
        "--split", type=str, default=None, choices=["test", "train", "val"]
    )

def diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    # 添加一个参数 --noise_schedule，用于指定噪声计划类型。默认值为 "cosine"，可选值为 "linear" 和 "cosine" 两种
    group.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=["linear", "cosine"],
        type=str,
        help="Noise schedule type",
    )
    # 添加一个参数 --diffusion_steps，用于指定扩散过程中的步数。默认值为 10
    group.add_argument(
        "--diffusion_steps",
        default=10,
        type=int,
        help="Number of diffusion steps (denoted T in the paper)",
    )
    # 添加一个参数 --timestep_respacing，用于指定时间步重分布的方式。默认值为 "ddim100"
    group.add_argument(
        "--timestep_respacing",
        default="ddim100",
        type=str,
        help="ddimN, else empty string",
    )
    # 添加一个参数 --sigma_small，用于指定是否使用较小的 sigma 值。默认值为 True
    group.add_argument(
        "--sigma_small", default=True, type=bool, help="Use smaller sigma values."
    )


def model_options(parser):
    group = parser.add_argument_group("model")

    # 定义了模型的层数，默认值为8，数据类型是整数
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")

    # 定义了音频处理层的数量，默认值为3，数据类型是整数
    group.add_argument(
        "--num_audio_layers", default=3, type=int, help="Number of audio layers."
    )

    # 定义了模型中的“heads”数量（例如，在多头注意力机制中），默认值为4，数据类型是整数
    group.add_argument("--heads", default=4, type=int, help="Number of heads.")

    # 定义了潜在维度（例如Transformer或GRU的宽度），默认值为512，数据类型是整数
    group.add_argument(
        "--latent_dim", default=512, type=int, help="Transformer/GRU width."
    )

    # 定义了在训练期间屏蔽条件的概率，默认值为0.20，数据类型是浮点数
    group.add_argument(
        "--cond_mask_prob",
        default=0.20,
        type=float,
        help="The probability of masking the condition during training."
        " For classifier-free guidance learning.",
    )

    # 定义了联合速度损失的权重，默认值为0.0，数据类型是浮点数
    group.add_argument(
        "--lambda_vel", default=0.0, type=float, help="Joint velocity loss."
    )

    # 定义了一个布尔型参数，通过 --unconstrained 选项来设置，表示模型是否不受任何文本或动作的约束。
    group.add_argument(
        "--unconstrained",
        action="store_true",
        help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
        "Currently tested on HumanAct12 only.",
    )

    # 定义了数据格式，可以选择“pose”或“face”，默认值为“pose”，数据类型是字符串
    group.add_argument(
        "--data_format",
        type=str,
        choices=["pose", "face"],
        default="pose",
        help="whether or not to use vae for diffusion process",
    )

    # 定义了一个布尔型参数，通过 --not_rotary 选项来设置，表示是否禁用旋转机制。
    group.add_argument("--not_rotary", action="store_true")

    # 定义了一个布尔型参数，通过 --simplify_audio 选项来设置，表示是否简化音频处理。
    group.add_argument("--simplify_audio", action="store_true")

    # 定义了一个参数 --add_frame_cond，数据类型是浮点数，可能的值只有1，默认值为None。
    group.add_argument("--add_frame_cond", type=float, choices=[1], default=None)





def training_options(parser):
    group = parser.add_argument_group("training")

    group.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="Path to save checkpoints and results.",
    )

    group.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, will enable to use an already existing save_dir.",
    )

    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
        help="Choose platform to log results. NoPlatform means no logging.",
    )

    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")

    group.add_argument(
        "--weight_decay", default=0.0, type=float, help="Optimizer weight decay."
    )

    group.add_argument(
        "--lr_anneal_steps",
        default=0,
        type=int,
        help="Number of learning rate anneal steps.",
    )

    group.add_argument(
        "--log_interval", default=1_000, type=int, help="Log losses each N steps"
    )

    group.add_argument(
        "--save_interval",
        default=5_000,
        type=int,
        help="Save checkpoints and run evaluation each N steps",
    )

    group.add_argument(
        "--num_steps",
        default=800_000,
        type=int,
        help="Training will stop after the specified number of steps.",
    )

    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="If not empty, will start from the specified checkpoint (path to model###.pt file).",
    )


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to results dir (auto created by the script). "
        "If empty, will create dir in parallel to checkpoint.",
    )
    group.add_argument("--face_codes", default=None, type=str)
    group.add_argument("--pose_codes", default=None, type=str)
    group.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Maximal number of prompts to sample, "
        "if loading dataset from file, this field will be ignored.",
    )
    group.add_argument(
        "--num_repetitions",
        default=3,
        type=int,
        help="Number of repetitions, per sample (text prompt/action)",
    )
    group.add_argument(
        "--guidance_param",
        default=2.5,
        type=float,
        help="For classifier-free sampling - specifies the s parameter, as defined in the paper.",
    )
    group.add_argument(
        "--curr_seq_length",
        default=None,
        type=int,
    )
    group.add_argument(
        "--render_gt",
        action="store_true",
        help="whether to use pretrained clipmodel for audio encoding",
    )


def add_generate_options(parser):
    group = parser.add_argument_group("generate")
    group.add_argument(
        "--plot",
        action="store_true",
        help="Whether or not to save the renderings as a video.",
    )
    group.add_argument(
        "--resume_trans",
        default=None,
        type=str,
        help="keyframe prediction network.",
    )
    group.add_argument("--flip_person", action="store_true")


def get_cond_mode(args):
    if args.dataset == "social":
        cond_mode = "audio"
    return cond_mode


def train_args():
    parser = ArgumentParser()
    base_options(parser)
    data_options(parser)
    diffusion_options(parser)
    model_options(parser)
    training_options(parser)
    return parser.parse_args()
