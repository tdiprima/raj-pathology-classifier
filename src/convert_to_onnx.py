import torch
import torch.onnx

model = models.resnet50(weights="IMAGENET1K_V1")

# Step 2: Load the saved weights
checkpoint = torch.load('../best_resnet.pth')
model.load_state_dict(checkpoint)  # or checkpoint['model_state_dict'] if it's a dict
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)

torch.onnx.export(model,
                  dummy_input,
                  "resnet.onnx",
                  export_params=True,
                  opset_version=11,  # or another version
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                               'output': {0: 'batch_size'}})

print("Model successfully converted to ONNX format!")
