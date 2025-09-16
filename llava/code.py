from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    # load_4bit=True   # 👈 important for quantization
)





import os
from PIL import Image
import matplotlib.pyplot as plt

save_folder = "/scratch/codepk37/anlpnerf/sam2/notebooks/mask_obj_figurines"

for file in os.listdir(save_folder):
    if not file.endswith((".png", ".jpg", ".jpeg")):
        continue
    
    img_path = os.path.join(save_folder, file)
    image = Image.open(img_path).convert("RGB")

    # Ask for short caption
    prompt = "Give a 2-3 word description of the main object."

    # Preprocess
    inputs = image_processor.preprocess(image, return_tensors="pt").to(model.device)
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(model.device)

    # Run model
    output_ids = model.generate(
        input_ids=input_ids,
        images=inputs["pixel_values"],
        max_new_tokens=20
    )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Show inline
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption, fontsize=12)
    plt.show()
