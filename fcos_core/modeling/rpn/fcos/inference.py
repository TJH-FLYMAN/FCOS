import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes

#这个文件主要完成推理阶段的操作，forward_for_single_feature_map对每个fpn level的结果做后处理。
#网络输出的box_cls, box_regression, centerness和locations作为输入，由pre_nms_thresh取阈值，
# 得到大于阈值的点的索引candidate_inds。
class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh #0.05
        self.pre_nms_top_n = pre_nms_top_n #1000
        self.nms_thresh = nms_thresh # 0.6
        self.fpn_post_nms_top_n = fpn_post_nms_top_n # 100
        self.min_size = min_size # 0
        self.num_classes = num_classes # 81

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W 
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape #推理N=1 ，训练N=batch

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1) #NHWC
        box_cls = box_cls.reshape(N, -1, C).sigmoid()  #预测的分类信息, (N, H*W,C)，sigmoid函数将输出值变为0到1之间
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)# shape (N,H,W,4)
        box_regression = box_regression.reshape(N, -1, 4)  #预测的回归信息shape (N,H*W,4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()  #预测的中心度数据 (N,H*W,1)

        candidate_inds = box_cls > self.pre_nms_thresh # bool型 0或1，论文中所述的大于0.05为正样本，其余为背景负样本
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)#sum(1)为对行求和，看一共有多少个正样本 ，为一个数值Tensor
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n) #pre_nms_top_n的值是否大于1000，大于1000取1000，小于保留原有不变。保留最多1000个正样本

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None] #训练的时候分类部分不和中心度相乘　测试的时候再相乘作为分类分数

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]# 拿出得分图中正样本的点
             # 得到正样本在得分图中的信息。得分图已经resize成（H*W,C），第一维相当于位置，第二维相当于类别。
            per_candidate_nonzeros = per_candidate_inds.nonzero()   
            per_box_loc = per_candidate_nonzeros[:, 0] #位置信息
            per_class = per_candidate_nonzeros[:, 1] + 1 # 类别信息 背景+1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]  # 每个正样本点对应的回归向量 举例Tensor(294,4)
            per_locations = locations[per_box_loc]  # 每个正样本点对应于原图像上的点 举例Tensor(294,2)

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():  # 正样本数是否大于1000个
                #选取前per_pre_nms_top_n个，不进行大小排序。返回一个元组 (values,indices)对应per_box_cls, top_k_indices
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices] # per_class保留的是所有正样本的类别信息，现在只取前1000个
                per_box_regression = per_box_regression[top_k_indices] # 回归向量，只取对应的前1000个
                per_locations = per_locations[top_k_indices] # 正样本点对应的原图像的点，只取前1000个

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)  # 预测的bbox的 x1,y1,x2,y2, 左上角顶点与右下角顶点

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)  # 将超出原图像的边界框进行平滑调整
            boxlist = remove_small_boxes(boxlist, self.min_size) # 将w和h为负数的去掉，此时的bbox还是 xyxy 形式的
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = [] # list:5,对应5层的不同特征，每一个都是BoxList类对象
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes)) # list：1，里面是5个元组
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                ) #NMS 
                num_labels = len(boxlist_for_class) #NMS后剩余BB数目
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0: #>100
                cls_scores = result.get_field("scores") #分类得分
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()  # bool型 0/1 大于 image_thresh的为1，其余为0
                keep = torch.nonzero(keep).squeeze(1) # 拿出keep中元素为1的索引
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH     #0.05
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N     #1000
    nms_thresh = config.MODEL.FCOS.NMS_TH               #0.6
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG #100

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh, #0.05
        pre_nms_top_n=pre_nms_top_n, #1000
        nms_thresh=nms_thresh, #0.6
        fpn_post_nms_top_n=fpn_post_nms_top_n, #100
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES #81
    )

    return box_selector
