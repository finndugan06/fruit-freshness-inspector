import gradio as gr
from collections import Counter
from ultralytics import YOLO

# Load the trained model
model = YOLO("weights/best.pt")


def inspect(image, conf):
    results = model.predict(image, conf=conf, verbose=False)
    r = results[0]

    # Draw boxes and labels; plot() returns BGR, so flip to RGB for display
    annotated = r.plot()[:, :, ::-1]

    # Count detections per class
    counts = Counter(model.names[int(c)] for c in r.boxes.cls.tolist())

    if not counts:
        return annotated, "No fruit detected. Try lowering the confidence threshold."

    lines = [f"{name}: {n}" for name, n in counts.items()]
    rotten = sum(n for name, n in counts.items() if name.startswith("rotten"))
    verdict = "REJECT: rotten fruit detected" if rotten else "PASS: all fruit fresh"

    return annotated, verdict + "\n\n" + "\n".join(lines)


demo = gr.Interface(
    fn=inspect,
    inputs=[
        gr.Image(type="pil", label="Upload a photo of apples or bananas"),
        gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(label="Detections"),
        gr.Textbox(label="Inspection result"),
    ],
    title="Fruit Freshness Inspector",
    description="YOLOv8n model trained to detect fresh and rotten apples and bananas.",
)

if __name__ == "__main__":
    demo.launch()
