"""
This module allows performing training on the ImageNet dataset with a MobileViT-S model in a
class-incremental learning setting consisting of 50 experiences of 20 classes each.
It performs the third experience starting from the second experience checkpoints produced by
`experiment28_imagenet_mobilevit_50_increments_20_classes_10400.py`.
"""

import os
import dill
import random
import math

from training.runner import Runner, RunnerPlugin
from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from benchmarks.classic import SplitImageNet
from training.storage_policy import ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from models.network_class_incremental import NetworkClassIncremental
from training.plugins import (
                              LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              TrainMBSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
                              )
from llifc.intervention import LLIFCDataCollection
from metrics.class_accuracy import incremental_classifier_class_accuracy_per_subnetwork_metrics
from metrics.confusion_matrix import incremental_classifier_confusion_matrix_per_subnetwork_metrics
from metrics.incremental_classifier_accuracy import incremental_classifier_accuracy_per_subnetwork_metrics
from metrics.incremental_classifier_loss import incremental_classifier_loss_per_subnetwork_metrics
from metrics.topk_accuracy import incremental_classifier_topk_accuracy_per_subnetwork_metrics
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import timing_metrics
from avalanche.logging import InteractiveLogger
random.seed(1277)  # set the random seed for the creation of the splits in SplitImageNet; TODO needs to be different
# for each experiment type

# Ensure reproducibility
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

EXP_ID = 28
"""
Experiment ID
"""

N_EXPS = 50
"""
Number of experiences.
"""

N_CLASSES_PER_EXP = 20
"""
Number of classes per experience
"""

N_TRAIN_SAMPLES_PER_CLASS = 1300
"""
Number of training samples per class
"""

N_EXEMPLARS = int(0.4 * N_TRAIN_SAMPLES_PER_CLASS * N_CLASSES_PER_EXP)
"""
The number of exemplars to keep in the replay buffer.
"""

N_CLASSES_TOTAL = N_EXPS * N_CLASSES_PER_EXP
"""
Total number of classes
"""

N_CLASS_ORDERS = 40
"""
Number of class orders
"""

ACTUAL_N_CLASS_ORDERS = 10
"""
The actual number of class orders
"""

CLASSES_1300 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 180, 182, 184, 185, 186, 187, 189, 191, 192, 193, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 267, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 391, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 410, 411, 412, 413, 414, 415, 416, 417, 419, 420, 421, 422, 423, 424, 425, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 482, 483, 484, 485, 486, 487, 488, 489, 490, 492, 493, 494, 495, 496, 497, 498, 500, 502, 504, 505, 506, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 522, 523, 524, 525, 526, 527, 528, 529, 530, 532, 533, 534, 535, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 568, 569, 570, 571, 572, 573, 574, 575, 576, 578, 579, 580, 581, 582, 584, 586, 587, 588, 589, 591, 592, 593, 594, 595, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 632, 633, 634, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 654, 655, 656, 657, 658, 659, 660, 661, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 677, 679, 680, 681, 682, 683, 684, 685, 687, 688, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 707, 709, 710, 711, 713, 715, 716, 717, 718, 719, 720, 721, 725, 726, 730, 732, 733, 734, 735, 736, 737, 738, 739, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 752, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 773, 774, 775, 776, 777, 778, 779, 780, 781, 783, 784, 785, 786, 787, 788, 790, 791, 792, 793, 794, 795, 796, 797, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 813, 814, 815, 816, 817, 818, 819, 820, 822, 823, 824, 825, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 839, 840, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 855, 856, 858, 859, 861, 862, 863, 864, 865, 866, 867, 868, 870, 871, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 886, 887, 888, 889, 890, 893, 894, 895, 896, 897, 898, 899, 900, 902, 903, 904, 905, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 922, 923, 924, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
"""
ImageNet class IDs with exactly 1300 samples
"""

CLASSES_NOT_1300 = [43, 51, 62, 98, 103, 147, 152, 158, 164, 165, 166, 167, 168, 175, 181, 183, 188, 190, 194, 206, 221, 252, 262, 268, 335, 390, 392, 409, 418, 426, 439, 465, 481, 491, 499, 501, 503, 507, 521, 531, 536, 550, 551, 567, 577, 583, 585, 590, 596, 610, 623, 630, 631, 635, 653, 662, 663, 675, 676, 678, 686, 689, 706, 708, 712, 714, 722, 723, 724, 727, 728, 729, 731, 740, 747, 753, 771, 772, 782, 789, 798, 810, 811, 812, 821, 826, 838, 841, 854, 857, 860, 869, 872, 885, 891, 892, 901, 906, 914, 921, 925, 926, 940, 946, 969]
"""
ImageNet class IDs with less than 1300 samples
"""

CLASS_ORDERS = [random.sample(CLASSES_1300, len(CLASSES_1300)) + random.sample(CLASSES_NOT_1300, len(CLASSES_NOT_1300)) for _ in range(N_CLASS_ORDERS)]
"""
The class orders that will be used for creating the splits in the SplitImageNet benchmark. Classes with less than 1300
samples always come after those with exactly 1300 samples.
"""

EXP_SEEDS = [[random.randint(0, 1300) for _ in range(N_EXPS)] for _ in range(N_CLASS_ORDERS)]
"""
The seeds that will be used to control the random operations during training for each different class order
"""

REHEARSAL_SEEDS = [random.randint(0, 1300) for _ in range(N_CLASS_ORDERS)]
"""
The seeds that will be used to create the rehearsal set for each different class order 
"""

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0),
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=63 / 255),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
"""
The transformations to be used during training.
"""

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(
        256,
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
"""
The transformations to be used during evaluation.
Obviously, they do not perform any sort of data augmentation.
"""

LR = 2.5e-4
"""
The initial learning rate for each experience.
"""

MINIMUM_LR = 1e-5
"""
Minimum learning rate for each experience
"""

WARMUP_EPOCHS = 5
"""
Number of warmup epochs
"""

WEIGHT_DECAY = 0.01
"""
The weight decay to be used for optimization.
"""

INIT_TRAIN_MB_SIZE = 128
"""
The training mini-batch size during the first experience 
"""

NEXT_TRAIN_MB_SIZE = 256
"""
The training mini-batch size during all experiences except for the first one. 128 samples are sampled from the current
experience's dataset while the remaining half is sampled from the replay buffer.
"""

EVAL_MB_SIZE = 1024
"""
The evaluation mini-batch size.
"""

TRAIN_EPOCHS = 100
"""
The number of training epochs.
"""

FREQUENCY = math.ceil((N_CLASSES_PER_EXP * N_TRAIN_SAMPLES_PER_CLASS * 0.9) / INIT_TRAIN_MB_SIZE)
"""
Frequency used by the Last-layer gradient scheduler
"""

CLASS_ORDERS = CLASS_ORDERS[:ACTUAL_N_CLASS_ORDERS]
EXP_SEEDS = EXP_SEEDS[:ACTUAL_N_CLASS_ORDERS]
REHEARSAL_SEEDS = REHEARSAL_SEEDS[:ACTUAL_N_CLASS_ORDERS]

class LinearCosineAnnealingLR(SequentialLR):
    def __init__(self, optimizer):
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_EPOCHS - WARMUP_EPOCHS,
            eta_min=MINIMUM_LR
        )
        super().__init__(optimizer=optimizer,
                         schedulers=[warmup_scheduler, cosine_scheduler],
                         milestones=[WARMUP_EPOCHS]
                         )

class ResizeBuffer(RunnerPlugin):
    def __init__(self, checkpoint: str, max_size: int, experience, save_path: str):
        super().__init__()
        ckp = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
        strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = ckp["strategy"]
        replay_plugin: ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = strategy.plugins[2]
        self.storage_policy = replay_plugin.storage_policy
        self.storage_policy.resize(None, max_size)
        self.experience = experience
        self.save_path = save_path

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        runner.strategy.plugins[2].storage_policy = self.storage_policy
        new_data = self.experience.dataset.train()
        new_data = new_data.update_data_attribute("targets_task_labels", [0 for _ in range(len(new_data))])
        self.storage_policy.update_from_dataset(new_data, None)
        buffer_datasets = self.storage_policy.buffer_datasets
        print(f"Buffer datasets: {[len(b) for b in buffer_datasets]}")
        torch.save(self.storage_policy, self.save_path, pickle_module=dill)


def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                    **kwargs):
    """
    Callback called after adapting the model to the current experience, updating the optimizers and setting up the
    training dataset for each subnetwork for the current experience.

    This callback creates a scheduler instance for the optimizer of each subnetwork.
    It casts each optimizer to :class`torch.optim.AdamW`
    :param strategy: a strategy
    :param kwargs: some keyword arguments to be passed
    """
    # creating a scheduler instance for each subnetwork's optimizer
    for sub_id, opt in strategy.optimizer.items():
        opt.__class__ = torch.optim.AdamW
        # create an instance of the scheduler and pass the respective optimizer and any provided keyword arguments
        self.scheduler_instncs[sub_id] = self.scheduler(optimizer=opt, **self.kwargs)
        if self.reset_lrs:
            # store the initial learning rates of each param group of the current optimizer in
            # `self._init_lrs[sub_id]`
            for group in opt.param_groups:
                self._init_lrs[sub_id].append(group['lr'])

LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.after_train_datasets_adaptation = after_train_datasets_adaptation

if __name__ == "__main__":

    for index, (class_order, exp_seeds, rehearsal_seed) in enumerate(zip(CLASS_ORDERS, EXP_SEEDS, REHEARSAL_SEEDS)):

        # If cuda is available, the default device is "cuda". Otherwise, "cpu" is used.
        # If this script is run using `torch.run` for distributed training, the device is returned by
        # `Runner.init_distributed`
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.environ.get("LOCAL_RANK", None) is not None:  # if LOCAL_RANK env var exists, then torch.run was used
            device = Runner.init_distributed()

        benchmark = SplitImageNet(
            n_experiences=N_EXPS,
            val_dataset=0.1,  # 10% of the samples in the train dataset are apportioned to validation
            seed=123,  # this seed is used to create the validation set so that to replicate results
            fixed_class_order=class_order,
            shuffle=False,
            train_transform=TRAIN_TRANSFORM,
            eval_transform=EVAL_TRANSFORM,
            dataset_root="/iridisfs/home/at2n19/hpc-shared-data/ImageNet/ImageNetILSVRC2012",
            include_indices=True,
            use_torchvision_ImageNet=False
        )

        strategy = DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
            # a single MobileViT-S model with a dynamic output layer for the class incremental setting
            model=NetworkClassIncremental(
                model="mobilevit_s",
                pretrained=False
            ),
            # the AdamW optimizer
            optimizer=AdamW(
                params=nn.Linear(1, 1).parameters(),  # just some dummy parameters
                lr=LR,
                weight_decay=WEIGHT_DECAY
            ),
            # since we only care about the incremental classifier, the outlier criterion weight is set to 0 so that
            # it does not influence the training process
            outlier_criterion_weight=0.,
            train_mb_size=INIT_TRAIN_MB_SIZE,
            train_epochs=TRAIN_EPOCHS,
            eval_mb_size=EVAL_MB_SIZE,
            device=device,
            plugins=[
                # During the first experience (experience 0), the training mini-batch size is 128. From the second
                # experience (including) onwards, the training mini-batch size is set to 256, i.e. 128 for the
                # samples in the current experience and 128 for the samples stored in the replay buffer.
                TrainMBSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                    train_mb_sizes={1: NEXT_TRAIN_MB_SIZE}
                ),
                LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                    scheduler=LinearCosineAnnealingLR,
                    step_calls="epoch",
                    reset_lrs=True
                ),
                # handles a replay memory buffer. The same number of samples are stored across each observed class.
                # The samples for each class are stored (uniform) randomly.
                ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                    storage_policy=ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                        max_size=N_EXEMPLARS,
                        # the first seed in `seeds` is used for seeding the creation of the replay buffer for each
                        # class
                        seed=rehearsal_seed
                    )
                ),
                LastLayerGradientInterventionsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                    intervention=LLIFCDataCollection(),
                    frequency=FREQUENCY,
                    excl_exps=[0]
                )
            ],
            evaluator=EvaluationPlugin(
                incremental_classifier_class_accuracy_per_subnetwork_metrics(experience=True),
                incremental_classifier_confusion_matrix_per_subnetwork_metrics(experience=True),
                incremental_classifier_accuracy_per_subnetwork_metrics(epoch=True, experience=True),
                incremental_classifier_loss_per_subnetwork_metrics(epoch=True, experience=True),
                incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=3, experience=True),
                timing_metrics(minibatch=True, epoch=True),
                loggers=[InteractiveLogger()]
            ),
            eval_every=1,
            peval_mode="epoch",
            do_initial=True,  # perform evaluation at the start of each experience before training starts
        )

        runner = Runner(strategy, benchmark)
        runner.run(
            save_checkpoint_path=f"experiment{EXP_ID}_imagenet_mobilevit_{N_EXPS}_increments_{N_CLASSES_PER_EXP}_classes_{index}_{N_EXEMPLARS}",
            load_checkpoint=f"experiment{EXP_ID}_imagenet_mobilevit_{N_EXPS}_increments_{N_CLASSES_PER_EXP}_classes_{index}_{N_EXEMPLARS}_experience_1.pth",
            device_checkpoint=torch.device("cpu"),
            seed=exp_seeds,
            experience_stop=3,
            eval_streams_type=["train", "val", "test"],
            plugins=[
                ResizeBuffer(
                    checkpoint=f"experiment{EXP_ID}_imagenet_mobilevit_{N_EXPS}_increments_{N_CLASSES_PER_EXP}_classes_{index}_{N_EXEMPLARS}_experience_0.pth",
                    max_size=N_EXEMPLARS * 2,
                    experience=benchmark.train_stream[1],
                    save_path=f"experiment{EXP_ID}_imagenet_mobilevit_{N_EXPS}_increments_{N_CLASSES_PER_EXP}_classes_{index}_{N_EXEMPLARS}_storage_policy_experience_1.pth"
                )
            ],
            num_workers=10,  # the number of workers used for dataloading
            reset_optimizer_state=True  # after each experience, the optimizer state is reset
        )
