import os
import cv2
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from copy import deepcopy
from pprint import pformat
from collections import OrderedDict
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as mse
from skimage.metrics import normalized_mutual_information as nmi
from scipy.stats import entropy as kl
from scipy.stats import wasserstein_distance as emd

from pytorch_diffusion import Diffusion
from utils.aggregate_block.model_trainer_generate import generate_cls_model

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()
transform_test = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
device_ids = [0,1]

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.module.named_children()]):
            raise ValueError("return_layers are not present in model") 
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.module.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break 
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
 
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

def pixel_similarity(img_a, img_b, method="ssim"):
    img_a = np.array(img_a)
    img_b = np.array(img_b)
    if method == "ssim":
        #similarity = ssim(img_a, img_b, multichannel=True) #[-1,1]
        similarity = (ssim(img_a, img_b, multichannel=True) + 1) * 0.5
    elif method == "mse":
        #similarity = mse(img_a, img_b) #[0,1] distance
        similarity = 1 - mse(img_a, img_b)
    elif method == "nmi":
        #similarity = nmi(img_a, img_b) #[1,2]
        similarity = nmi(img_a, img_b) - 1
    elif method == "hist":
        sub_imga = cv2.split(img_a)
        sub_imgb = cv2.split(img_b)
        similarity = 0.0
        for imga, imgb in zip(sub_imga, sub_imgb):
            hista = cv2.calcHist([imga], [0], None, [32], [0.0, 256.0])
            histb = cv2.calcHist([imgb], [0], None, [32], [0.0, 256.0])
            similarity += cv2.compareHist(hista, histb, cv2.HISTCMP_CORREL)
        similarity = similarity / 3 #[0,1]
    return similarity

def PearsonCorrelation(feature_a,feature_b):
    x = feature_a
    y = feature_b
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pearsoncorrelation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearsoncorrelation

def feature_similarity(feature_a, feature_b, method="cosine"):
    if method == "cosine":
        #similarity = torch.cosine_similarity(feature[0],feature[1],dim=0) #[-1,1]
        similarity = (torch.cosine_similarity(feature_a,feature_b,dim=0).item() + 1) * 0.5
    elif method == "pdist":
        #similarity = torch.pairwise_distance(feature[0],feature[1])  #p=1,2 distance [0,max]
        similarity = 1 / (torch.pairwise_distance(feature_a,feature_b).item() + 1)
    elif method == "pearson":
        #similarity = PearsonCorrelation(feature[0],-1 * feature[0]) #[-1,1]
        similarity = (PearsonCorrelation(feature_a,feature_b).item() + 1) * 0.5
    return similarity

def distance(img_a, img_b, feature_a, feature_b, lamda=0.5, method_pixel="ssim", method_feature="cosine"):
    pixel_similar = pixel_similarity(img_a, img_b, method_pixel)
    feature_similar = feature_similarity(feature_a, feature_b, method_feature)
    pixel_distance = 1 - pixel_similar
    feature_distance = 1 - feature_similar
    img_distance = lamda * pixel_distance + (1 - lamda) * feature_distance
    return img_distance

def tensor2img(tensorslist):
    imgslist = [[] for i in range(len(tensorslist))]
    for i in range(len(tensorslist)):
        for j in range(len(tensorslist[i])):
            tensor = tensorslist[i][j]
            tensor = torch.clamp(tensor, -1, 1)
            tensor = unnormalize_to_zero_to_one(tensor)
            tensor = torch.clamp(tensor, 0, 1)
            img = topil(tensor)
            imgslist[i].append(img)
    return imgslist

def suspicious(model, datalist, min_num, device="cuda"):
    sus_data = [[] for i in range(len(datalist))]
    sus_target = [[] for i in range(len(datalist))]
    softmax = nn.Softmax(dim=-1)
    for index, imgs in enumerate(datalist):
        tensors = []
        for j in range(len(imgs)):
            tensors.append(transform_test(imgs[j]))
        tensors = torch.stack(tensors).to(device)
        with torch.no_grad():
            prob = model(tensors)
            prob, labels = torch.max(softmax(prob), -1)
        labels = labels.detach().cpu().numpy()
        labels_denoised = labels[1:]
        counter = Counter(labels_denoised)
        if len(set(labels_denoised)) == 1:
            sus_target[index].extend(labels.tolist())
            sus_data[index].extend(imgs)
        else:
            if counter.most_common(2)[1][1] >= min_num:
                sus_target[index].extend(labels.tolist())
                sus_data[index].extend(imgs)
            else:
                sus_target[index].append(labels[0])
                sus_data[index].append(imgs[0])
                for j in range(len(labels_denoised)):
                    if labels_denoised[j] == counter.most_common(1)[0][0]:
                        sus_data[index].append(imgs[j+1])
                        sus_target[index].append(counter.most_common(1)[0][0])
    return sus_data, sus_target

def get_poisonlabel(suspicious_target, denoised_dataset, method_dist="kl", method_outlier="mad"):
    poison_label = 0
    labels = [[] for i in range(len(set(denoised_dataset.targets)))]
    for i in range(len(suspicious_target)):
        if len(set(suspicious_target[i])) != 1:
            labels[denoised_dataset.targets[i]].extend(suspicious_target[i])
    metric = []
    if method_dist == "count":
        for i in range(len(labels)):
            metric.append(len(labels[i]))
    else:
        for i in range(len(labels)):
            p = np.array([0 for j in range(len(labels))])
            p[i] = 1
            q = np.array([labels[i].count(j) / len(labels[i]) for j in range(len(labels))])
            if method_dist == "kl":
                metric.append(kl(p, q))
            elif method_dist == "js":
                M = (p + q) / 2
                metric.append(0.5 * kl(p, M) + 0.5 * kl(q, M))
            elif method_dist == "emd":
                metric.append(emd(p,q))
            elif method_dist == "bd":
                metric.append(-np.log(np.sum(np.sqrt(p*q))))
    logging.info(f"metric: {metric}")
    metric = np.array(metric)
    if method_outlier == "mad":
        median = np.median(metric)
        mad = np.median(np.abs(metric - median))
        score = np.abs(metric - median)/1.4826/(mad + 1e-6)
        logging.info(f"score: {score}")
        poison_label = np.where(score > 2)[0] 
    elif method_outlier == "ipq":
        q1 = np.quantile(metric, 0.25)
        q3 = np.quantile(metric, 0.75)
        iqr = q3 - q1
        up_bound = q3 + 1.5 * iqr
        poison_label = np.where(metric > up_bound)[0]
    elif method_outlier == "3sigma":
        mean = np.mean(metric)
        std = np.std(metric)
        up_bound = mean + 2.5 * std
        poison_label = np.where(metric > up_bound)[0]
    logging.info(f"poison label: {poison_label}")
    return poison_label

def get_pseudolabel(model, denoised_dataset, device='cuda'):
    labels = []
    softmax = nn.Softmax(dim=-1)
    for i in tqdm(range(len(denoised_dataset.data))):
        imgs = denoised_dataset.data[i][0]
        tensors = []
        for j in range(len(imgs)):
            tensors.append(transform_test(imgs[j]))
        tensors = torch.stack(tensors).to(device)
        with torch.no_grad():
            prob = model(tensors)
            prob, label = torch.max(softmax(prob), -1)
        label = label.detach().cpu().numpy()
        labels.append(label)
    return np.array(labels)

def getfeature(model, suspicious_data, denoised_dataset, poison_label, layer_name="layer4", device="cuda"):
    suspicious_features = []
    model = IntermediateLayerGetter(model, {layer_name:'feature'})
    for i in tqdm(range(len(suspicious_data))):
        if denoised_dataset.targets[i] in poison_label:
            tensors = []
            for j in range(len(suspicious_data[i])):
                tensors.append(transform_test(suspicious_data[i][j]))
            tensors = torch.stack(tensors).to(device)
            with torch.no_grad():
                out = model(tensors)
                feature = out["feature"]
                feature = adaptive_avg_pool2d(feature, output_size=(1, 1))
                feature = feature.squeeze(3).squeeze(2).cpu()
            suspicious_features.append(feature)
        else:
            feature = torch.zeros([1,1])
            suspicious_features.append(feature)
    return suspicious_features

def getimg(imgs, features):
    dists = [0]
    for i in range(1, len(imgs)):
        dist = distance(imgs[0], imgs[i], features[0], features[i])
        dists.append(dist)
    sorted_index = np.argsort(np.array(dists))
    return sorted_index

def sus2clean(suspicious_data, suspicious_target, denoised_dataset, model, args):
    denoised_data = []
    denoised_target = []
    poison_label = get_poisonlabel(suspicious_target, denoised_dataset)
    suspicious_features = getfeature(model, suspicious_data, denoised_dataset, poison_label, layer_name="layer4") 
    torch.save(suspicious_features, "../results/" + args.attack+ "/suspicious_features.pt")
    suspicious_features = torch.load("../results/" + args.attack+ "/suspicious_features.pt")
    for i in tqdm(range(len(suspicious_data))):
        if len(set(suspicious_target[i])) == 1:
            denoised_data.append(suspicious_data[i][0])
            denoised_target.append(denoised_dataset.targets[i])
        else:
            if len(set(suspicious_target[i][1:])) == 1:
                if denoised_dataset.targets[i] in poison_label:
                    denoised_data.append(suspicious_data[i][getimg(suspicious_data[i],suspicious_features[i])[int(len(suspicious_data[i])*0.8)]])
                    denoised_target.append(suspicious_target[i][1])
                else:
                    denoised_data.append(suspicious_data[i][0])
                    denoised_target.append(denoised_dataset.targets[i])
            else:
                if denoised_dataset.targets[i] in poison_label:
                    denoised_data.append([suspicious_data[i],suspicious_features[i]])
                    denoised_target.append(Counter(suspicious_target[i][1:]).most_common(1)[0][0])
                else:
                    denoised_data.append(suspicious_data[i][0])
                    denoised_target.append(denoised_dataset.targets[i])
    return denoised_data, np.array(denoised_target), poison_label

def diffusion_purification(denoised_dataset, batch_size, model=None, mode="train", min_num=5, t=150, max_iter=5, per_num=10):
    raw_dataloader = DataLoader(dataset=denoised_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    suspicious_data = []
    suspicious_target = []
    diffusion = Diffusion.from_pretrained("cifar10")
    for batch_idx, (x, *additional_info) in tqdm(enumerate(raw_dataloader)):
        x = normalize_to_neg_one_to_one(x) #[-1,1]
        tensorslist = [[] for i in range(len(x))]
        for i in range(len(tensorslist)):
            tensorslist[i].append(x[i])
        with torch.no_grad():
            for iter in range(max_iter):
                x = diffusion.diffuse_t_steps(x, t)
                x, tensorlist = diffusion.denoise(x.shape[0],n_steps=t,x=x.to(diffusion.device),curr_step=t)
                for i in range(len(tensorslist)):
                    tensorslist[i].extend(tensorlist[i][len(tensorlist[i])-per_num:])
                x = torch.clamp(x, -1, 1)
        datalist = tensor2img(tensorslist)
        if mode == "train":
            sus_data, sus_target = suspicious(model, datalist, min_num=min_num)
            suspicious_data.extend(sus_data)
            suspicious_target.extend(sus_target)
        elif mode == "test":
            suspicious_data.extend(datalist)
    return suspicious_data, suspicious_target

def test(model, test_dataset, batch_size, device):
    test_data = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    metrics = {'test_correct': 0,'test_total': 0}
    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in tqdm(enumerate(test_data)):
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()
            metrics['test_correct'] += correct.item()
            metrics['test_total'] += target.size(0)
    logging.info(f"test accuracy: {metrics['test_correct'] / metrics['test_total']}")

def average_dataset(raw_dataset, index_clean, suspicious_data, poison_label):
    denoised_dataset = deepcopy(raw_dataset)
    counter = Counter(denoised_dataset.targets)
    logging.info(pformat(counter))
    max_num = counter.most_common(1)[0][1]
    samples_count = [0 for i in range(len(set(denoised_dataset.targets)))]
    samples_num = [[] for i in range(len(set(denoised_dataset.targets)))]
    for key, value in counter.items():
        if key not in poison_label:
            count = [0 for i in range(value)]
        else:
            count = [int(max_num/value) for i in range(value)]
            remain_count = max_num - value * int(max_num / value)
            if remain_count > 0:
                index = random.sample([i for i in range(value)], remain_count)
                for i in range(value):
                    if i in index:
                        count[i] += 1
        samples_num[key].extend(count)
    denoised_data = denoised_dataset.data
    denoised_target = denoised_dataset.targets.tolist()
    original_targets = denoised_dataset.original_targets.tolist()
    poison_indicator = denoised_dataset.poison_indicator.tolist()
    original_index = denoised_dataset.original_index.tolist()
    for i in tqdm(range(len(denoised_dataset.data))):
        img_num = samples_num[denoised_dataset.targets[i]][samples_count[denoised_dataset.targets[i]]]
        if img_num > 0:
            index = index_clean[i]
            imgs = suspicious_data[index]
            for j in range(img_num - 1):
                denoised_data.append(random.choice(imgs[1:]))
                denoised_target.append(denoised_dataset.targets[i])
                original_targets.append(denoised_dataset.original_targets[i])
                poison_indicator.append(denoised_dataset.poison_indicator[i])
                original_index.append(denoised_dataset.original_index[i])
        samples_count[denoised_dataset.targets[i]] += 1
    random_index = np.arange(0,len(denoised_target),1)
    np.random.shuffle(random_index)
    denoised_dataset.data = np.array(denoised_data, dtype=object)[random_index].tolist()
    denoised_dataset.targets = np.array(denoised_target)[random_index]
    denoised_dataset.original_targets = np.array(original_targets)[random_index]
    denoised_dataset.poison_indicator = np.array(poison_indicator)[random_index]
    denoised_dataset.original_index = np.array(original_index)[random_index]
    logging.info(pformat(Counter(denoised_dataset.targets)))
    return denoised_dataset

def purify(raw_dataset, batch_size=1024, mode="train", classifier=None, adv_dataset=None, poison_label=None, args=None): #num_diffusion_timesteps=1000
    denoised_dataset = deepcopy(raw_dataset)
    if mode == "train":
        model_state_dict = torch.load(os.getcwd()+"/../record/" + args.attack + "_preresnet/attack_epoch_99.pt")
        classifier = generate_cls_model(args.model,args.num_classes)
        classifier.to(args.device)
        classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)
        classifier.load_state_dict(model_state_dict["model_state_dict"])        
        classifier.eval()
        suspicious_data, suspicious_target = diffusion_purification(denoised_dataset, model=classifier, batch_size=batch_size, mode=mode)
        np.save("../results/" + args.attack+ "/suspicious_data.npy", np.array(suspicious_data, dtype=object))
        np.save("../results/" + args.attack+ "/suspicious_target.npy", np.array(suspicious_target, dtype=object))
        suspicious_data = np.load("../results/" + args.attack+ "/suspicious_data.npy",allow_pickle=True).tolist()
        suspicious_target = np.load("../results/" + args.attack+ "/suspicious_target.npy",allow_pickle=True).tolist()
        denoised_data, denoised_target, poison_label = sus2clean(suspicious_data, suspicious_target, denoised_dataset, classifier, args)
        index_clean = [i for i in range(len(denoised_data)) if not isinstance(denoised_data[i], list)]
        index_suspicious = [i for i in range(len(denoised_data)) if isinstance(denoised_data[i], list)]
        denoised_dataset.data = denoised_data
        denoised_dataset.targets = denoised_target 
        suspicious_dataset = deepcopy(denoised_dataset)
        denoised_dataset.subset(index_clean)
        suspicious_dataset.subset(index_suspicious)
        averaged_dataset = average_dataset(denoised_dataset, index_clean, suspicious_data, poison_label)
        logging.info(f"clean dataset labels correct rate: {len(np.where(denoised_dataset.targets == denoised_dataset.original_targets)[0])}, {len(index_clean)}")
        poison_index = np.where(denoised_dataset.poison_indicator==1)[0]
        logging.info(f"clean dataset poison samples labels correct rate: {len(np.where(denoised_dataset.targets[poison_index] == denoised_dataset.original_targets[poison_index])[0])}, {len(poison_index)}")
        logging.info(f"clean dataset poison samples undetect: {len(np.where(denoised_dataset.targets[poison_index] == int(args.attack_target))[0])}")  #multi label attack
        logging.info(f"suspicious dataset samples num: {len(suspicious_dataset.targets)}, clean samples num: {len(np.where(suspicious_dataset.poison_indicator==0)[0])}, poison samples num: {len(np.where(suspicious_dataset.poison_indicator==1)[0])}")
        return denoised_dataset, suspicious_dataset, index_clean, index_suspicious, averaged_dataset, poison_label

    elif mode == "test":
        suspicious_data, _ = diffusion_purification(denoised_dataset, batch_size, mode=mode)
        np.save("../results/" + args.attack+ "/test_data.npy", np.array(suspicious_data, dtype=object))
        suspicious_data = np.load("../results/" + args.attack+ "/test_data.npy",allow_pickle=True).tolist()
        denoised_data = []
        denoised_target = []
        original_targets = []
        poison_indicator = []
        original_index = []
        for i in tqdm(range(len(suspicious_data))):
            for j in range(1, len(suspicious_data[i])):
                denoised_data.append(suspicious_data[i][j])
                denoised_target.append(denoised_dataset.targets[i])
                original_targets.append(denoised_dataset.original_targets[i])
                poison_indicator.append(denoised_dataset.poison_indicator[i])
                original_index.append(denoised_dataset.original_index[i])
        denoised_dataset.data = denoised_data
        denoised_dataset.targets = np.array(denoised_target)
        denoised_dataset.original_targets = np.array(original_targets)
        denoised_dataset.poison_indicator = np.array(poison_indicator)
        denoised_dataset.original_index = np.array(original_index)
        return raw_dataset, denoised_dataset

    elif mode == "pseudo":
        if classifier == None:
            model_state_dict = torch.load(os.getcwd()+"/../record/" + args.attack + "_purify/attack_epoch_99.pt")
            classifier = generate_cls_model(args.model,args.num_classes)
            classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)
            classifier.load_state_dict(model_state_dict["model_state_dict"])     
        classifier.to(args.device)
        classifier.eval()
        labels = get_pseudolabel(classifier, denoised_dataset)
        targets = np.array([labels[i][0] for i in range(len(labels))])
        clean_index = np.where(denoised_dataset.poison_indicator==0)[0]
        poison_index = np.where(denoised_dataset.poison_indicator==1)[0]
        logging.info(f"suspicious dataset clean samples pseudo label correct rate: {len(np.where(targets[clean_index] == denoised_dataset.original_targets[clean_index])[0])}, {len(clean_index)}")
        logging.info(f"suspicious dataset poison samples pseudo label correct rate: {len(np.where(targets[poison_index] == denoised_dataset.original_targets[poison_index])[0])}, {len(poison_index)}")
        logging.info(f"suspicious dataset poison samples pseudo label undetect: {len(np.where(targets[poison_index] == int(args.attack_target))[0])}")
        train_index = [i for i in range(len(targets)) if targets[i] == denoised_dataset.targets[i]]
        denoised_dataset.subset(train_index)
        labels = labels[train_index]
        clean_index = np.where(denoised_dataset.poison_indicator==0)[0]
        poison_index = np.where(denoised_dataset.poison_indicator==1)[0]
        logging.info(f"pseudo dataset clean samples pseudo label correct rate: {len(np.where(denoised_dataset.targets[clean_index] == denoised_dataset.original_targets[clean_index])[0])}, {len(clean_index)}")
        logging.info(f"pseudo dataset poison samples pseudo label correct rate: {len(np.where(denoised_dataset.targets[poison_index] == denoised_dataset.original_targets[poison_index])[0])}, {len(poison_index)}")
        logging.info(f"pseudo dataset poison samples pseudo label undetect: {len(np.where(denoised_dataset.targets[poison_index] == int(args.attack_target))[0])}")
        counter_denoised = Counter(denoised_dataset.targets)
        logging.info(pformat(counter_denoised))  # label numbers not same, poison label need to be trained
        clean_dataset = deepcopy(adv_dataset)
        counter_clean = Counter(clean_dataset.targets)
        logging.info(pformat(counter_clean))
        max_num = 0
        for key, value in counter_clean.items():
            max_num = max(value + counter_denoised[key], max_num)
        logging.info(max_num)
        samples_count = [0 for i in range(len(set(denoised_dataset.targets)))]
        samples_num = [[] for i in range(len(set(denoised_dataset.targets)))]
        for key, value in counter_clean.items():
            if key not in poison_label:
                count = [1 for j in range(counter_denoised[key])]
            else:
                count = [int((max_num-value)/counter_denoised[key]) for j in range(counter_denoised[key])]
                remain_count = max_num - value - counter_denoised[key] * int((max_num-value)/counter_denoised[key])
                index = random.sample([j for j in range(counter_denoised[key])], remain_count)
                for j in range(counter_denoised[key]):
                    if j in index:
                        count[j] += 1
            samples_num[key].extend(count)
        denoised_data = clean_dataset.data
        denoised_target = clean_dataset.targets.tolist()
        original_targets = clean_dataset.original_targets.tolist()
        poison_indicator = clean_dataset.poison_indicator.tolist()
        original_index = clean_dataset.original_index.tolist()
        for i in tqdm(range(len(denoised_dataset.data))):
            imgs = [denoised_dataset.data[i][0][0]]
            features = [denoised_dataset.data[i][1][0]]
            for j in range(1, len(labels[i])):
                if labels[i][j] == denoised_dataset.targets[i]:
                    imgs.append(denoised_dataset.data[i][0][j])
                    features.append(denoised_dataset.data[i][1][j])
            img_num = min(int(len(imgs)*0.8), samples_num[denoised_dataset.targets[i]][samples_count[denoised_dataset.targets[i]]])
            sorted_index = getimg(imgs,features)
            for j in range(img_num):
                denoised_data.append(imgs[sorted_index[int(len(imgs)*0.8)-j]])
                denoised_target.append(denoised_dataset.targets[i])
                original_targets.append(denoised_dataset.original_targets[i])
                poison_indicator.append(denoised_dataset.poison_indicator[i])
                original_index.append(denoised_dataset.original_index[i])
            samples_count[denoised_dataset.targets[i]] += 1
        random_index = np.arange(0,len(denoised_target),1)
        np.random.shuffle(random_index)
        clean_dataset.data = np.array(denoised_data, dtype=object)[random_index].tolist()
        clean_dataset.targets = np.array(denoised_target)[random_index]
        clean_dataset.original_targets = np.array(original_targets)[random_index]
        clean_dataset.poison_indicator = np.array(poison_indicator)[random_index]
        clean_dataset.original_index = np.array(original_index)[random_index]
        logging.info(pformat(Counter(clean_dataset.targets)))
        return clean_dataset
        
def label_and_save(raw_dataset, denoised_dataset, suspicious_dataset, index_clean, index_suspicious, classifier=None, args=None):
    if classifier == None:
        model_state_dict = torch.load(os.getcwd()+"/../record/" + args.attack + "_purify/finetune_epoch_best.pt")
        classifier = generate_cls_model(args.model,args.num_classes)
        classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)
        classifier.load_state_dict(model_state_dict["model_state_dict"])        
    classifier.to(args.device)
    classifier.eval()
    labels = get_pseudolabel(classifier, suspicious_dataset)
    targets = np.array([labels[i][0] for i in range(len(labels))]) 
    clean_index = np.where(suspicious_dataset.poison_indicator==0)[0]
    poison_index = np.where(suspicious_dataset.poison_indicator==1)[0]
    logging.info(f"suspicious dataset clean samples label correct rate: {len(np.where(targets[clean_index] == suspicious_dataset.original_targets[clean_index])[0])}, {len(clean_index)}")
    logging.info(f"suspicious dataset poison samples label correct rate: {len(np.where(targets[poison_index] == suspicious_dataset.original_targets[poison_index])[0])}, {len(poison_index)}")
    logging.info(f"suspicious dataset poison samples label undetect: {len(np.where(targets[poison_index] == int(args.attack_target))[0])}")
    suspicious_dataset.targets = targets
    data = []
    for i in tqdm(range(len(suspicious_dataset.data))):
        imgs = [suspicious_dataset.data[i][0][0]]
        features = [suspicious_dataset.data[i][1][0]]
        for j in range(1, len(labels[i])):
            if labels[i][j] == targets[i]:
                imgs.append(suspicious_dataset.data[i][0][j])
                features.append(suspicious_dataset.data[i][1][j])
        if len(imgs) == 1:
            data.append(imgs[0])
        else:
            data.append(imgs[getimg(imgs,features)[int(len(imgs)*0.8)]])
    suspicious_dataset.data = data
    final_dataset = deepcopy(raw_dataset)
    final_dataset.targets[index_clean] = denoised_dataset.targets
    final_dataset.targets[index_suspicious] = suspicious_dataset.targets
    final_data = np.array(final_dataset.data, dtype=object)
    final_data[index_clean] = np.array(denoised_dataset.data, dtype=object)
    final_data[index_suspicious] = np.array(suspicious_dataset.data, dtype=object)
    final_dataset.data = final_data.tolist()
    real_poison_index = np.where(final_dataset.poison_indicator==1)
    pred_poison_index = np.where(final_dataset.targets!=raw_dataset.targets)
    pred_correct_poison_index = np.intersect1d(real_poison_index, pred_poison_index)
    pred_wrong_poison_index = np.setdiff1d(pred_poison_index, pred_correct_poison_index)
    logging.info(f"real poison samples: {len(real_poison_index[0])}, pred poison samples: {len(pred_poison_index[0])}")
    logging.info(f"pred detect rate: {len(pred_correct_poison_index) / len(real_poison_index[0])}")
    logging.info(f"pred false positive rate: {len(pred_wrong_poison_index) / len(np.where(final_dataset.poison_indicator==0)[0])}")        
    logging.info(f"pred relabel correct rate of poison samples: {(final_dataset.original_targets[pred_correct_poison_index]==final_dataset.targets[pred_correct_poison_index]).sum() / len(pred_correct_poison_index)}")
    logging.info(f"labels total correct rate: {len(np.where(final_dataset.original_targets==final_dataset.targets)[0]) / len(final_dataset.targets)}")  
    np.save("../results/" + args.attack+ "/clean_data.npy", np.array(final_dataset.data, dtype=object))
    np.save("../results/" + args.attack+ "/clean_target.npy", np.array(final_dataset.targets, dtype=object))  
    logging.info(f"save finished") 