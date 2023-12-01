import torch
from torch import nn
import torch.nn.functional as F
from helpers import get_device

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=3):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, teacher_outputs):
        soft_target_loss = nn.functional.kl_div(
            nn.functional.log_softmax(outputs / self.temperature, dim=1),
            nn.functional.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        hard_target_loss = self.ce_loss(outputs, targets)

        total_loss = soft_target_loss + hard_target_loss
        return total_loss

def relu_evidence(y):
    relu = torch.nn.ReLU()
    return relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def proposed_kl_divergence(predicted_alpha, true_alpha, num_class, device=None):
    if not device:
        device = get_device()
    predicted_sum_alpha = torch.sum(predicted_alpha, dim=1, keepdim=True)
    true_sum_alpha = torch.sum(true_alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(predicted_sum_alpha)
        - torch.lgamma(predicted_alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(true_alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(true_alpha.sum(dim=1, keepdim=True))
    )
    second_term = (
        (predicted_alpha - true_alpha)
        .mul(torch.digamma(predicted_alpha) - torch.digamma(predicted_sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def ce_loss(p, output, num_class, global_step, annealing_step, device):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=num_class)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    enhanced_evidence = torch.zeros(output.shape[0], num_class).to(device)
    for i in range(output.shape[0]):
        evidence_sum = torch.sum(evidence[i])
        enhanced_evidence[i][torch.nonzero(label[i]).item()] = evidence_sum
    #B = annealing_coef * kl_divergence(alp, num_class, device)
    B = annealing_coef * proposed_kl_divergence(alpha, enhanced_evidence+1, num_class, device)
    return torch.mean((A + B))

def loglikelihood_loss(y, alpha, num_classes, global_step, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    y = F.one_hot(y, num_classes = num_classes)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    e = alpha - 1
    alp = e * (1 - y) + 1
    annealing_coef = min(1, global_step / annealing_step)
    B = annealing_coef * kl_divergence(alp, num_classes, device)
    
    return loglikelihood 


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss
