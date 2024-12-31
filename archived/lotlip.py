import argparse

import torch
from PIL import Image

import open_clip


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Where to store files cache_dir.",
    )


    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "lotlip_bert-ViT-B-16",
        pretrained="model.pt",
        cache_dir=args.cache_dir,  # Use the cache directory
    )
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer("lotlip_bert-ViT-B-16")


    # Step 2: Define images and text prompt
    image_paths = ["figures/puppy.png", "figures/cat.png", "figures/moti.png"]
    text_prompt = "a beautiful cat that looks like a dog"


    # Preprocess images and tokenize text
    images = [preprocess(Image.open(img_path).convert("RGB")) for img_path in image_paths]
    image_tensor = torch.stack(images)  # Shape: [N, 3, H, W]
    text_tokens = tokenizer([text_prompt])


    # Step 3: Generate embeddings
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)  # Shape: [1, D]
        image_embeddings = model.encode_image(image_tensor)  # Shape: [N, D]


    # Step 4: Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(text_embedding, image_embeddings) # Shape: [N]


    # Step 5: Select the most appropriate image
    best_match_idx = similarities.argmax().item()
    best_image_path = image_paths[best_match_idx]


    # print("Encoded Image:", image_embeddings)
    # print("Encoded Text:", text_embedding)
    print(f"Most appropriate image: {best_image_path}")


if __name__ == "__main__":
    main()
