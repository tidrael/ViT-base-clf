from transformers import ViTFeatureExtractor, ViTForImageClassification


class ImageClassifier:
    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )

    def predict(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        class_name = self.model.config.id2label[predicted_class_idx]
        return str(class_name).title()
