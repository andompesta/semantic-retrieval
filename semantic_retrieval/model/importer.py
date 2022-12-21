import clip
from semantic_retrieval.model import CLIP
from semantic_retrieval.model.config import LargeClipConfig
import torch
from PIL import Image
from pathlib import Path

if __name__ == "__main__":
    base_path = Path.cwd()
    checkpoints_path = base_path.joinpath("checkpoints")
    device = "cpu"

    model, preprocess = clip.load(
        "ViT-L/14@336px",
        device=device,
        download_root=checkpoints_path.joinpath("original").as_posix(),
    )

    collector = []
    for layer_name, params in model.state_dict().items():
        print("{} \t {}".format(layer_name, params.size()))

        # adjust visual transformer names
        if layer_name.startswith("visual."):
            layer_name = layer_name.replace("visual.", "vision.")
            if "transformer" in layer_name:
                layer_name = layer_name.replace("transformer.", "")
        elif layer_name.startswith("transformer."):
            layer_name = layer_name.replace("transformer.", "text.")
        

        if layer_name == "vision.proj":
            collector.append(("image_projection", params))

        elif layer_name == "vision.class_embedding":
            collector.append(("vision.cls_token", params))
        elif layer_name == "vision.positional_embedding":
            collector.append(("vision.pos_embedding", params))

        elif layer_name == "token_embedding.weight":
            collector.append(("text.token_embedding.weight", params))
        elif layer_name == "positional_embedding":
            collector.append(("text.pos_embedding", params))
        elif layer_name == "ln_final.weight":
            collector.append(("text.ln_final.weight", params))
        elif layer_name == "ln_final.bias":
            collector.append(("text.ln_final.bias", params))

        else:
            collector.append((layer_name, params))

    my_model_config = LargeClipConfig()
    my_model = CLIP(**my_model_config.to_dict())

    my_model.load_state_dict(dict(collector))

    image = preprocess(
        Image.open("data/raw/shoose-boss.png")
    ).unsqueeze(0).to(device)
    text = clip.tokenize([
        "a balck shose",
        "a shoose",
        "boss clothing",
    ]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        my_image_features = my_model.encode_image(image)
        my_text_features = my_model.encode_text(text)

        assert (image_features - my_image_features).sum() == 0
        assert (text_features - my_text_features).sum() == 0

    torch.save(
        my_model.to("cpu").state_dict(),
        checkpoints_path.joinpath("ViT-L-14@336px.pt").as_posix(),
    )
