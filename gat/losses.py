import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_loss(x, y):
    # 1 - cosine similarity
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return 1 - (x * y).sum(dim=1).mean()

def compute_migg_loss(
    positive_feature, negative_feature, sketch_feature,
    gcn_processed_label_features,
    prediction_scores: torch.Tensor,
    true_labels_multihot: torch.Tensor,
    model_parameters: torch.nn.Parameter = None, # For potential regularization
) -> tuple[torch.Tensor, dict]:
    # Loss = L_CrossEntropy + L_TripletLoss + L_Semantic
    
    loss_weights = {'cross_entropy': 0.3, 'triplet': 0.3, 'semantic': 0.3, 'regularization': 0.1}

    individual_losses = {}
    
    # L_CrossEntropy
    criterion_bce = nn.BCEWithLogitsLoss()
    loss_ce = criterion_bce(prediction_scores, true_labels_multihot.float())
    individual_losses['cross_entropy'] = loss_ce.item()
    
    # L_TripletLoss
    criterion_triplet = nn.TripletMarginLoss(margin=0.3)
    loss_triplet = criterion_triplet(sketch_feature, positive_feature, negative_feature)
    individual_losses['triplet'] = loss_triplet.item()
    
    # L_Semantic
    loss_sem = 0
    num_labels = true_labels_multihot.sum(dim=1).clamp(min=1)  # tránh chia cho 0

    for i in range(gcn_processed_label_features.size(0)):  # với mỗi nhãn
        mask = true_labels_multihot[:, i] > 0
        if mask.sum() > 0:
            sem = gcn_processed_label_features[i].unsqueeze(0)  # (1, 64)
            sem = sem.expand(mask.sum(), -1)      # match shape
            loss_sem += cosine_loss(sketch_feature[mask], sem)
            loss_sem += cosine_loss(positive_feature[mask], sem)
            loss_sem += cosine_loss(negative_feature[mask], sem)

    loss_sem = loss_sem / num_labels.sum()
    individual_losses['semantic'] = loss_sem.item()
    
    # --- Optional: L_regularization (e.g., L2 weight decay on model parameters) ---
    # This is often handled by the optimizer's `weight_decay` parameter.
    # However, if explicit L2 reg is needed as a loss term:
    loss_reg = torch.tensor(0.0, device=prediction_scores.device)
    if model_parameters is not None and loss_weights.get('regularization', 0.1) > 0:
        l2_reg = torch.tensor(0., device=prediction_scores.device)
        for param in model_parameters:
            if param.requires_grad: # Ensure we only penalize learnable parameters
                l2_reg += torch.norm(param, p=2)
        loss_reg = l2_reg
    individual_losses['regularization'] = loss_reg.item() * loss_weights.get('regularization', 0.0)
    
    total_loss = (
        loss_weights.get('cross_entropy', 0.3) * loss_ce +
        loss_weights.get('triplet', 0.3) * loss_triplet +
        loss_weights.get('semantic', 0.3) * loss_sem +
        loss_weights.get('regularization', 0.1) * loss_reg # Add weighted regularization
    )
    individual_losses['total'] = total_loss.item()
    
    return total_loss, individual_losses

if __name__ == "__main__":
        # Thiết lập thông số
    batch_size = 4
    feature_dim = 64  # Giống với sketch_feature, positive_feature, etc.
    num_classes = 10
    gcn_dim = 300      # Input dim cho Linear(300 -> 64)

    # Tạo dữ liệu giả
    sketch_feature = torch.randn(batch_size, feature_dim, requires_grad=True)
    positive_feature = torch.randn(batch_size, feature_dim, requires_grad=True)
    negative_feature = torch.randn(batch_size, feature_dim, requires_grad=True)

    gcn_processed_label_features = torch.randn(num_classes, gcn_dim, requires_grad=True)
    prediction_scores = torch.randn(batch_size, num_classes, requires_grad=True)
    true_labels_multihot = torch.randint(0, 2, (batch_size, num_classes)).float()

    # Giả lập model parameters (nếu cần regularization)
    model = nn.Linear(64, num_classes)  # ví dụ thôi
    model_parameters = list(model.parameters())

    # Gọi hàm loss
    total_loss, loss_dict = compute_migg_loss(
        positive_feature=positive_feature,
        negative_feature=negative_feature,
        sketch_feature=sketch_feature,
        gcn_processed_label_features=gcn_processed_label_features,
        prediction_scores=prediction_scores,
        true_labels_multihot=true_labels_multihot,
        model_parameters=model_parameters
    )

    # In kết quả
    for k, v in loss_dict.items():
        print(f"{k}: {v}")