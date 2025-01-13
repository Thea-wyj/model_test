```
h5
python explain_cmd.py --model_path model/res_net/ResNet_Model.h5 --input_dir data/signal/testdatas/part1.h5 --method_name GRAD  --data_type h5 --class_num 24
python explain_cmd.py --model_path model/res_net/ResNet_Model.h5 --input_dir data/signal/testdatas/part1.h5 --method_name SG --data_type h5 --class_num 24
python explain_cmd.py --model_path model/res_net/ResNet_Model.h5 --input_dir data/signal/testdatas/part1.h5 --method_name IG --data_type h5 --class_num 24

csv
python explain_cmd.py --model_path model/ecg_model.hdf5 --input_dir data/ECG200/ECG200_TEST.CSV --method_name GRAD --data_type csv --class_num 2
python explain_cmd.py --model_path model/ecg_model.hdf5 --input_dir data/ECG200/ECG200_TEST.CSV --method_name SG   --data_type csv --class_num 2
python explain_cmd.py --model_path model/ecg_model.hdf5 --input_dir data/ECG200/ECG200_TEST.CSV --method_name IG   --data_type csv --class_num 2
```