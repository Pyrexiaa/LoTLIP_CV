from ultralytics import YOLO


def yolo(image_url):
    model = YOLO("yolov8n.pt")
    results = model(image_url)

    output = []

    for result in results:
        for box in result.boxes.data.tolist():
            x_min, y_min, x_max, y_max, confidence, class_id = box
            output.append({
                "image": image_url.split("/")[-1],
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "class_id": int(class_id),
                "confidence": confidence,
            })

    return output

if __name__ == "__main__":
    # image_url = "https://ultralytics.com/images/bus.jpg"
    image_url = "figures/cat.png"
    output = yolo(image_url)
    print(output)
