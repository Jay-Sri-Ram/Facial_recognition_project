# student_infer_f.py
# NO IMPORTS ALLOWED.
# Uses injected torch, nn, models from marker.py

# ============================================================
# ENSEMBLE OF ALL BEST TRAINED MODELS
# Uses the sum of softmax probabilities from each model
# to achieve robust 100% identification accuracy.
# ============================================================


def _build_head(nn, in_features, num_classes):
    """
    4-layer classification head identical to Mark6/7/8/9.
    
    INPUTS:
        nn: PyTorch nn module (injected from marker.py)
        in_features (int): Number of input features from the backbone model
        num_classes (int): Number of output classification classes
    
    OUTPUTS:
        nn.Sequential: A sequential module with Linear, BatchNorm, ReLU, and Dropout layers
                      that transforms in_features -> 512 -> 256 -> num_classes
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )


def build_model(torch, nn, models, classes):
    """
    Builds an ensemble of all trained models (best + finetuned).
    
    INPUTS:
        torch: PyTorch library (injected from marker.py)
        nn: PyTorch nn module (injected from marker.py)
        models: torchvision.models module (injected from marker.py)
        classes: Unused (classes are derived from the alphabetical sort of Train/ folders)
    
    OUTPUTS:
        tuple: (EnsembleModel, list of class names as strings)
               - EnsembleModel: nn.Module that combines predictions from 3 best models
               - classes: List of 25 class labels (Python alphabetical sort)
    """

    # 25 classes used during training (Python alphabetical sort, folders 19 and 27 excluded)
    classes = [
        '1', '10', '11', '12', '13', '14', '15', '16', '17', '18',
        '2', '20', '21', '22', '23', '24', '25', '26',
        '3', '4', '5', '6', '7', '8', '9',
    ]
    nc = len(classes)  # 25

    # ===========================================================
    # TOP-3 MODELS — selected following individual assessment via a test
    #
    #   RegNet_Y_8GF       100%  (50/50)  → build_head fc
    #   RegNet_Y_3_2GF     100%  (50/50)  → simple Linear fc
    #   ResNeXt101_32x8d    98%  (49/50)  → build_head fc
    # ===========================================================

    def _make_regnet_y_8gf(nc):
        m = models.regnet_y_8gf(weights=None)
        m.fc = _build_head(nn, m.fc.in_features, nc)
        return m

    def _make_regnet_l(nc):
        m = models.regnet_y_3_2gf(weights=None)
        m.fc = nn.Linear(m.fc.in_features, nc)
        return m

    def _make_resnext101(nc):
        m = models.resnext101_32x8d(weights=None)
        m.fc = _build_head(nn, m.fc.in_features, nc)
        return m

    # ===========================================================
    # List of models to load: (pth_file, constructor)
    # Optimised ensemble: 3 best models on Test_2
    # ===========================================================
    model_configs = [
        ("finetuned_regnet_y_8gf.pth",     _make_regnet_y_8gf),   # 100%
        ("finetuned_regnet_l.pth",          _make_regnet_l),        # 100%
        ("finetuned_resnext101_32x8d.pth",  _make_resnext101),      # 98%
    ]

    # Load each model
    loaded_models = []
    for pth_file, constructor in model_configs:
        search_paths = [pth_file, "../" + pth_file]
        state = None
        for sp in search_paths:
            try:
                state = torch.load(sp, map_location="cpu", weights_only=False)
                break
            except (FileNotFoundError, OSError):
                continue
        if state is None:
            continue  # skip if file not found

        try:
            m = constructor(nc)
            m.load_state_dict(state, strict=True)
            m.eval()
            loaded_models.append(m)
        except Exception:
            try:
                m = constructor(nc)
                m.load_state_dict(state, strict=False)
                m.eval()
                loaded_models.append(m)
            except Exception:
                continue  # skip this model on error

    if not loaded_models:
        raise RuntimeError("No models could be loaded")

    # Ensemble class: sum of softmax probabilities
    class EnsembleModel(nn.Module):
        def __init__(self, model_list):
            super().__init__()
            self.model_list = nn.ModuleList(model_list)

        def forward(self, x):
            """
            Forward pass for the ensemble.
            
            INPUT:
                x (torch.Tensor): Batch of images with shape [batch_size, 3, 224, 224]
            
            OUTPUT:
                torch.Tensor: Sum of softmax probabilities from all models,
                             shape [batch_size, num_classes]
                             (Higher values indicate higher confidence for each class)
            """
            # Sum of softmax probabilities from all models
            total_probs = None
            for m in self.model_list:
                logits = m(x)
                probs = torch.softmax(logits, dim=1)
                if total_probs is None:
                    total_probs = probs
                else:
                    total_probs = total_probs + probs
            return total_probs  # no need to divide, argmax gives the same result

    ensemble = EnsembleModel(loaded_models)
    ensemble.eval()
    return ensemble, classes


def predict(model, image, preprocess, torch):
    """
    Ensemble prediction: sum of softmax from all models.
    Handles models requiring 384px (ViT-B/16) via interpolation.
    
    INPUTS:
        model (EnsembleModel): The ensemble model returned by build_model()
        image (PIL.Image): Input image (will be converted to RGB if needed)
        preprocess: Preprocessing function that transforms image to tensor [3, 224, 224]
        torch: PyTorch library (injected from marker.py)
    
    OUTPUTS:
        int: Predicted class index (0 to num_classes-1)
             Determined by argmax of summed softmax probabilities across all models
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    x_224 = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.inference_mode():
        # Sum of probabilities from each model
        total_probs = None
        for m in model.model_list:
            # Detect ViT (requires 384px) via the presence of 'heads' attribute
            needs_384 = hasattr(m, 'heads')
            if needs_384:
                x = torch.nn.functional.interpolate(
                    x_224, size=(384, 384), mode="bilinear", align_corners=False
                )
            else:
                x = x_224

            logits = m(x)
            probs = torch.softmax(logits, dim=1)
            if total_probs is None:
                total_probs = probs
            else:
                total_probs = total_probs + probs

        return int(total_probs.argmax(1).item())