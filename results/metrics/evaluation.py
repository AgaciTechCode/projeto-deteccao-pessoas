from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Avaliamos apenas BBOX (caixas) para evitar o erro de segmentação que você teve
evaluator = COCOEvaluator("person_test", output_dir="./output", tasks=("bbox",))
val_loader = build_detection_test_loader(cfg, "person_test")

print("--- MÉTRICAS DE DESEMPENHO ---")
results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(results)
