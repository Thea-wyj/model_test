```
python interpret_lime_test.py --model_path model/roberta-base-finetuned-chinanews-chinese --output_dir output/ --gpu 
python interpret_grads_occ_test.py --model_path model/roberta-base-finetuned-chinanews-chinese --output_dir output/  --saliency occlusion  --gpu 
python interpret_grads_occ_test.py --model_path model/roberta-base-finetuned-chinanews-chinese --output_dir output/  --saliency deeplift  --gpu 
python interpret_grads_occ_test.py --model_path model/roberta-base-finetuned-chinanews-chinese --output_dir output/  --saliency ig  --gpu  
python interpret_shap_test.py --model_path model/roberta-base-finetuned-chinanews-chinese --output_dir output/ --gpu 

```