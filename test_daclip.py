import torch
from PIL import Image
import open_clip

checkpoint = 'models/epoch_30.pt'
model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
image = Image.open(r"E:\dataset\MSRS_bad_weather_val_new\raindrop\moderately\01226N.png")
for i in range(len(preprocess.transforms)):
    image = preprocess.transforms[i](image)
image = image.unsqueeze(0)
degradations = ['foggy lightly','foggy moderately','foggy heavily','rainy lightly',
                'rainy moderately','rainy heavily','snowy lightly','snowy moderately','snowy heavily',
                'raindrop lightly','raindrop moderately','raindrop heavily','common']
text = tokenizer(degradations)
task_name = 'fog'
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    image_features, degra_features = model.encode_image(image, control=True)
    degra_features /= degra_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * degra_features @ text_features.T).softmax(dim=-1)
    index = torch.argmax(text_probs[0])

print(f"Task: {task_name}: {degradations[index]} - {text_probs[0][index]}")