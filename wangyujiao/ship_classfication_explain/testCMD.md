```
python explain_cmd.py --model_path pretrainedModel.pth --input_dir data/test/HM/876.jpg --method_name GradientShap 
python explain_cmd.py --model_path pretrainedModel.pth --input_dir data/test/HM/876.jpg --method_name IntegratedGradients 
python explain_cmd.py --model_path pretrainedModel.pth --input_dir data/test/HM/876.jpg --method_name Occlusion 
python explain_cmd.py --model_path pretrainedModel.pth --input_dir data/test/HM/876.jpg --method_name GuidedGradCam 
python explain_cmd.py --model_path pretrainedModel.pth --input_dir data/test/HM/876.jpg --method_name Lime 
```