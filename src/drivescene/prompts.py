"""One prompt contract for supervised training and inference; no ground truth context."""


def build_prompt(task: str, question: str, camera_names: list[str]) -> str:
    header = "\n".join(f"{cam}: <image>" for cam in camera_names)
    return (
        "You are analyzing a driving scene for offline research. "
        "Answer the question directly using the camera images. "
        "If the visual evidence is insufficient, state uncertainty.\n"
        f"Task: {task}\n{header}\nQuestion: {question}"
    )


def multimodal_content(sample: dict, images: list) -> list[dict]:
    """Replace only our known header markers; question text is never interpreted as markup."""
    cameras = sample["camera_names"]
    if len(cameras) != len(images) or not images:
        raise ValueError("Camera names and images must have the same nonzero length")
    prompt = build_prompt(sample["task"], sample["question"], cameras)
    # Split the prefix and camera header from the question to avoid treating text as images.
    prefix, _ = prompt.split("\n", 1)
    content = [{"type": "text", "text": prefix + f"\nTask: {sample['task']}\n"}]
    for cam, image in zip(cameras, images):
        content.extend([{"type": "text", "text": f"{cam}: "}, {"type": "image", "image": image}, {"type": "text", "text": "\n"}])
    content.append({"type": "text", "text": f"Question: {sample['question']}"})
    return content
