import foolbox
import foolbox.attacks as fa
import argparse


# 创建foolbox attack method 对象
def createMethodObj(config_dict):
    method_name = config_dict.get("method_name")
    if method_name == "L2ContrastReductionAttack":
        return "L2", fa.L2ContrastReductionAttack(target=config_dict.get("target", 0.5))
        # target relative to  the bounds from 0 to 1 (max)

    if method_name == "VirtualAdversarialAttack":
        return "L2", fa.VirtualAdversarialAttack(steps=config_dict.get("steps", 100), xi=config_dict.get("xi", 1e-6))

    if method_name == "DDNAttack":
        return "L2", fa.DDNAttack(init_epsilon=config_dict.get("init_epsilon", 1.0),
                                  steps=config_dict.get("steps", 100),
                                  gamma=config_dict.get("gamma", 0.05))

    elif method_name == "L2PGD":
        abs_stepsize = None if config_dict.get("abs_stepsize", "None") == "None" else config_dict.get("abs_stepsize")
        return "L2", fa.L2PGD(rel_stepsize=config_dict.get("rel_stepsize", 0.025), abs_stepsize=abs_stepsize,
                              steps=config_dict.get("steps", 50), random_start=(config_dict.get("random_start",
                                                                                                "True") == "True"))
    elif method_name == "LinfPGD":
        abs_stepsize = None if config_dict.get("abs_stepsize", "None") == "None" else config_dict.get("abs_stepsize")
        return "Linf", fa.LinfPGD(rel_stepsize=config_dict.get("rel_stepsize", 0.01 / 0.3), abs_stepsize=abs_stepsize,
                                  steps=config_dict.get("steps", 40),
                                  random_start=(config_dict.get("random_start", "True")
                                                == "True"))

    elif method_name == 'L2BasicIterativeAttack' or method_name == 'LinfBasicIterativeAttack':
        abs_stepsize = None if config_dict.get("abs_stepsize", "None") == "None" else config_dict.get("abs_stepsize")
        if method_name == 'L2BasicIterativeAttack':
            return "L2", fa.L2BasicIterativeAttack(rel_stepsize=config_dict.get("rel_stepsize", 0.2),
                                                   abs_stepsize=abs_stepsize,
                                                   steps=config_dict.get("steps", 10),
                                                   random_start=(config_dict.get("random_start",
                                                                                 "False") == "False"))
        elif method_name == 'LinfBasicIterativeAttack':
            return "Linf", fa.LinfBasicIterativeAttack(rel_stepsize=config_dict.get("rel_stepsize", 0.2),
                                                       abs_stepsize=abs_stepsize,
                                                       steps=config_dict.get("steps", 10),
                                                       random_start=(config_dict.get("random_start",
                                                                                     "False") == "False"))
    if method_name == "L2FastGradientAttack":
        return "L2", fa.FGM(random_start=config_dict.get("random_start", 'False') != 'False')
    elif method_name == "LinfFastGradientAttack":
        return "Linf", fa.FGSM(random_start=config_dict.get("random_start", 'False') != 'False')

    elif method_name == "L2AdditiveGaussianNoiseAttack":
        return "L2", fa.L2AdditiveGaussianNoiseAttack()
    elif method_name == "L2AdditiveUniformNoiseAttack":
        return "L2", fa.L2AdditiveUniformNoiseAttack()

    elif method_name == "L2ClippingAwareAdditiveGaussianNoiseAttack":
        return "L2", fa.L2ClippingAwareAdditiveGaussianNoiseAttack()
    elif method_name == "L2ClippingAwareAdditiveUniformNoiseAttack":
        return "L2", fa.L2ClippingAwareAdditiveUniformNoiseAttack()

    elif method_name == "LinfAdditiveUniformNoiseAttack":
        return "Linf", fa.LinfAdditiveUniformNoiseAttack()

    elif method_name == "L2RepeatedAdditiveGaussianNoiseAttack":
        return "L2", fa.L2RepeatedAdditiveGaussianNoiseAttack(repeats=config_dict.get("repeats", 100),
                                                              check_trivial=(config_dict.get("check_trivial",
                                                                                             "True") == "True"))
    elif method_name == "L2RepeatedAdditiveUniformNoiseAttack":
        return "L2", fa.L2RepeatedAdditiveUniformNoiseAttack(repeats=config_dict.get("repeats", 100),
                                                             check_trivial=(config_dict.get("check_trivial",
                                                                                            "True") == "True"))
    elif method_name == "L2ClippingAwareRepeatedAdditiveUniformNoiseAttack":
        return "L2", fa.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(repeats=config_dict.get("repeats", 100),
                                                                          check_trivial=(
                                                                                  config_dict.get("check_trivial",
                                                                                                  "True") == "True"))
    elif method_name == "L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack":
        return "L2", fa.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(repeats=config_dict.get("repeats", 100),
                                                                           check_trivial=(
                                                                                   config_dict.get("check_trivial",
                                                                                                   "True") == "True"))
    elif method_name == "LinfRepeatedAdditiveUniformNoiseAttack":
        return "Linf", fa.LinfRepeatedAdditiveUniformNoiseAttack(repeats=config_dict.get("repeats", 100),
                                                                 check_trivial=(config_dict.get(
                                                                     "check_trivial", "True") == "True"))
    elif method_name == 'L2CarliniWagnerAttack':
        # binary_search_steps=9, steps=10000, stepsize=0.01, confidence=0, initial_const=0.001, abort_early=True
        return "L2", fa.L2CarliniWagnerAttack(binary_search_steps=config_dict.get("binary_search_steps", 9),
                                              steps=config_dict.get("steps", 10000),
                                              stepsize=config_dict.get("stepsize", 0.01),
                                              confidence=config_dict.get("confidence", 0),
                                              initial_const=config_dict.get("initial_const", 0.001),
                                              abort_early=config_dict.get("abort_early", "True") == "True")

    elif method_name == 'NewtonFoolAttack':
        return "L2", fa.NewtonFoolAttack(steps=config_dict.get('steps', 100),
                                         stepsize=config_dict.get('stepsize', 0.01))

    elif method_name == "GaussianBlurAttack":
        return "None", fa.GaussianBlurAttack(distance=None, steps=1000, channel_axis=None, max_sigma=None)

    elif method_name in ["LinfDeepFoolAttack", "L2DeepFoolAttack"]:
        # steps=50, candidates=10, overshoot=0.02, loss='logits'
        if method_name == "LinfDeepFoolAttack":
            return "Linf", fa.LinfDeepFoolAttack(steps=config_dict.get('steps', 50),
                                                 candidates=config_dict.get('candidates', 10),
                                                 overshoot=config_dict.get('overshoot', 0.02),
                                                 loss=config_dict.get('loss', 'logits'))

        elif method_name == "L2DeepFoolAttack":
            return "L2", fa.L2DeepFoolAttack(steps=config_dict.get('steps', 50),
                                             candidates=config_dict.get('candidates', 10),
                                             overshoot=config_dict.get('overshoot', 0.02),
                                             loss=config_dict.get('loss', 'logits'))

    elif method_name == "SaltAndPepperNoiseAttack":
        return "None", fa.SaltAndPepperNoiseAttack()

    elif method_name == 'BoundaryAttack':
        # init_attack=None if config_dict.get("init_attack", 'None') == 'None'
        return "", fa.BoundaryAttack(init_attack=fa.LinfDeepFoolAttack(),
                                     steps=config_dict.get("steps", 25000),
                                     spherical_step=config_dict.get("spherical_step", 0.01),
                                     source_step=config_dict.get('source_step', 0.01),
                                     source_step_convergance=config_dict.get('source_step_convergance', 1e-07),
                                     step_adaptation=config_dict.get('step_adaptation', 1.5),
                                     update_stats_every_k=config_dict.get('update_stats_every_k', 10))
    elif method_name == 'HopSkipJumpAttack':
        return config_dict.get("constraint"), fa.hop_skip_jump.HopSkipJumpAttack(init_attack=fa.LinfDeepFoolAttack(),
                                                                                 steps=config_dict.get("steps", 64),
                                                                                 initial_gradient_eval_steps=config_dict.get(
                                                                                     "initial_gradient_eval_steps",
                                                                                     100),
                                                                                 max_gradient_eval_steps=config_dict.get(
                                                                                     "max_gradient_eval_steps", 10000),
                                                                                 stepsize_search=config_dict.get(
                                                                                     "stepsize_search",
                                                                                     "geometric_progression"),
                                                                                 gamma=config_dict.get("gamma", 1.0),
                                                                                 constraint=config_dict.get(
                                                                                     "constraint", "l2")
                                                                                 )
    return None


# str two bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# 解析命令行参数
def parseCmdArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for Attack')
    parser.add_argument('--targeted', type=str2bool, nargs='+', default=[False, False, False, False, False],
                        help='perform targeted attack ?')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size, the size of sample to use generate '
                                                                   'adversarial examples')
    parser.add_argument('--model_url', type=str,
                        default='http://localhost:9090/model/20230921/151816-mlp_fashionmnist.h5',
                        help='model url')
    # parser.add_argument('--model_url', type=str,
    #                     default='http://10.105.240.103:9000/model/20230404/17324-LeNet_fashionmnist.pt',
    #                     help='model url')  # pytorch
    parser.add_argument('--dataset_url', type=str,
                        default='http://localhost:9090/dataset/20230921/151750-fashion-mnist.zip',
                        help='dataset url')
    # parser.add_argument('--epsilons', type=float, nargs='+', default=[0.001, 0.001, 0.001, 0.001, 0.001],
    # help='epsilon list') parser.add_argument('--methodList', type=str, nargs='+', default=["L2FastGradientAttack",
    # "LinfFastGradientAttack", "LinfPGD", "L2BasicIterativeAttack", "LinfBasicIterativeAttack"], help='method name
    # list')
    parser.add_argument('--configFilePath', type=str,
                        default=r'D:\pythonDemo\alg\config\eea961e7-7195-4367-a7fe-3be287459d2b.json',
                        help='method config file path')

    parser.add_argument('--debug', type=int, default=1, help='debug ? true for 1 false for 0')
    # 解析参数
    args = parser.parse_args()
    return args


method_name_map = {
    "L2FastGradientAttack": "快速梯度方法（二范数）",
    "LinfFastGradientAttack": "快速梯度符号攻击方法",
    "LinfPGD": "投影梯度下降算法（无穷范数）",
    "L2PGD": "投影梯度下降算法（二范数）",
    "L2BasicIterativeAttack": "基础迭代攻击（二范数）",
    "LinfBasicIterativeAttack": "基础迭代攻击",
    "L2CarliniWagnerAttack": "Carlini & Wagner攻击（二范数）",
    "L2DeepFoolAttack": "DeepFool",
    "LinfDeepFoolAttack": "DeepFool（二范数）",
    "NewtonFoolAttack": "NewtonFool Attack",
    "BoundaryAttack": "Decision-Based Adversarial Attacks",
    "HopSkipJumpAttack": "Hop Skip Jump Attack"
}
