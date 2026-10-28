"""Source-only reconstruction + adversarial alignment + orthogonality."""
import torch
from torch import nn, optim

class LossFunctions:

    def __init__(self, beta=1.0, gamma=1.0):
        self.beta = beta
        self.gamma = gamma
        self.cross_entropy_loss = nn.BCELoss()

    def reconstruction_loss(self, e_original, e_reconstructed):
        if e_original.size(1) != e_reconstructed.size(1):
            e_reconstructed = e_reconstructed.view(e_original.size(0), -1)
        return torch.mean((e_original - e_reconstructed) ** 2)

    def domain_adversarial_loss(self, domain_pred, domain_labels):
        return self.cross_entropy_loss(domain_pred.squeeze(), domain_labels.float().squeeze())

    def orthogonality_loss(self, h_shared_source, h_related_source, h_shared_target, h_related_target):
        h_shared_source = nn.functional.normalize(h_shared_source, dim=1)
        h_related_source = nn.functional.normalize(h_related_source, dim=1)
        h_shared_target = nn.functional.normalize(h_shared_target, dim=1)
        h_related_target = nn.functional.normalize(h_related_target, dim=1)
        loss_source = torch.norm(torch.mm(h_shared_source.t(), h_related_source), p='fro') ** 2
        loss_target = torch.norm(torch.mm(h_shared_target.t(), h_related_target), p='fro') ** 2
        return (loss_source + loss_target) / h_shared_source.size(1)

def pretrain_model(X_source, X_target, representation_model, reconstruction_network, domain_discriminator, loss_functions, epochs=100):

    # 定义优化器，使用 Adam 优化器，学习率为 0.001
    optimizer = optim.Adam(list(representation_model.parameters()) +
                           list(reconstruction_network.parameters()) +
                           list(domain_discriminator.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 开始循环
    for epoch in range(epochs):
        # 梯度清零
        optimizer.zero_grad()
        # 得到各相关特征
        h_source_related, h_target_related, h_source_shared, h_target_shared = representation_model(
            X_source, X_target)
        reconstructed_source = reconstruction_network(h_source_shared, h_source_related)
        reconstructed_target = reconstruction_network(h_target_shared, h_target_related)

        # 计算重建损失
        loss_reconstruction_source = loss_functions.reconstruction_loss(X_source, reconstructed_source)
        loss_reconstruction_target = loss_functions.reconstruction_loss(X_target, reconstructed_target)

        domain_pred_source = domain_discriminator(h_source_shared)
        domain_pred_target = domain_discriminator(h_target_shared)

        domain_labels_source = torch.zeros(X_source.size(0), dtype=torch.float32)
        domain_labels_target = torch.ones(X_target.size(0), dtype=torch.float32)

        loss_domain = loss_functions.domain_adversarial_loss(
            torch.cat([domain_pred_source, domain_pred_target]),
            torch.cat([domain_labels_source, domain_labels_target])
        )

        # 正交性损失
        loss_orth = loss_functions.orthogonality_loss(
            h_source_shared, h_source_related,
            h_target_shared, h_target_related
        )

        total_loss = loss_reconstruction_source + loss_reconstruction_target + loss_functions.beta * loss_domain + loss_functions.gamma * loss_orth

        print(
            f"Epoch [{epoch}/{epochs}], "
            f"重构损失: {loss_reconstruction_source + loss_reconstruction_target:.4f}, "
            f"对抗损失: {loss_domain:.4f}, "
            f"正交损失: {loss_orth:.4f}"
        )

        total_loss.backward()       # 反向传播计算梯度

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(representation_model.parameters()) +
            list(reconstruction_network.parameters()) +
            list(domain_discriminator.parameters()),
            max_norm=1.0  # 最大梯度范数限制为1.0
        )

        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Pre-training Loss: {total_loss.item():.4f}')

